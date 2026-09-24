"""Counterfactual labels, controller fitting, and full-session evaluation."""

import json
import math
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .controller import ForgettingController
from .runtime import CBFSession


def _ids(value, name):
    if not isinstance(value, list) or any(not isinstance(x, int) or x < 0 for x in value):
        raise ValueError(f"{name} must be a list of nonnegative token IDs")
    return value


def load_scenarios(path: str, split: str) -> list[dict]:
    """Read pretokenized scenarios and reject source groups shared across splits."""
    scenarios = []
    group_splits = {}
    ids_seen = set()
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["id"]
            group = row["group_id"]
            row_split = row["split"]
            if not isinstance(sample_id, str) or not isinstance(group, str) or not sample_id or not group:
                raise ValueError(f"id and group_id must be nonempty strings at line {line_number}")
            if row_split not in ("train", "dev", "test"):
                raise ValueError(f"invalid split at line {line_number}")
            if sample_id in ids_seen:
                raise ValueError(f"duplicate scenario id: {sample_id}")
            ids_seen.add(sample_id)
            if group in group_splits and group_splits[group] != row_split:
                raise ValueError(f"group {group} occurs in more than one split")
            group_splits[group] = row_split
            _ids(row["context_ids"], "context_ids")
            if not row["futures"]:
                raise ValueError(f"scenario {sample_id} needs at least one future")
            for future in row["futures"]:
                _ids(future.get("continuation_ids", []), "continuation_ids")
                if not future["queries"]:
                    raise ValueError(f"scenario {sample_id} has a future without queries")
                for query in future["queries"]:
                    if not _ids(query["query_ids"], "query_ids") or not _ids(query["answer_ids"], "answer_ids"):
                        raise ValueError("query_ids and answer_ids must be nonempty")
            if row_split == split:
                scenarios.append(row)
    if not scenarios:
        raise ValueError(f"no scenarios for split {split}")
    return scenarios


def parse_grid(raw: str) -> list[float]:
    grid = sorted(set(float(part) for part in raw.split(",")))
    if not grid or grid[0] != 0.0 or grid[-1] != 1.0 or any(not math.isfinite(x) for x in grid):
        raise ValueError("the candidate grid must be finite, lie in [0,1], and include 0 and 1")
    return grid


def load_controller(path: str, model, device) -> ForgettingController:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(checkpoint["layers"]) != tuple(sorted(model.config.ttt_layers)):
        raise ValueError("controller TTT layers do not match the model")
    if checkpoint["hidden_size"] != model.config.hidden_size:
        raise ValueError("controller hidden size does not match the model")
    if checkpoint["scalar_size"] != 4 + 3 * len(model.config.ttt_layers):
        raise ValueError("controller scalar feature count does not match the model")
    controller = ForgettingController(
        checkpoint["hidden_size"], checkpoint["scalar_size"], checkpoint["width"], checkpoint["semantic_size"]
    )
    controller.load_state_dict(checkpoint["state_dict"])
    return controller.to(device).eval()


def _choose_alpha(session, policy, semantic, scalars):
    if policy == "baseline":
        return 0.0
    if policy == "controller":
        if session.controller is None:
            raise ValueError("controller state policy requires --controller")
        with torch.inference_mode():
            return session.controller.predict(semantic, scalars).item()
    return float(policy)


def collect_labels(model, scenarios: list[dict], output: str, grid: list[float],
                   state_policy: str = "baseline", controller=None, every: int = 1,
                   tie_tolerance: float = 1e-6) -> int:
    if every < 1 or tie_tolerance < 0:
        raise ValueError("every must be positive and tie_tolerance nonnegative")
    count = 0
    with open(output, "w", encoding="utf-8") as sink:
        for scenario in scenarios:
            session = CBFSession(model, controller)
            context = scenario["context_ids"]
            full_length = len(context) - len(context) % session.chunk_size
            for start in range(0, full_length, session.chunk_size):
                boundary = start // session.chunk_size + 1
                selected = boundary % every == 0
                if selected:
                    device = session.device
                    on_cuda = device.type == "cuda"
                    if on_cuda:
                        torch.cuda.synchronize(device)
                        start_allocated = torch.cuda.memory_allocated(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    else:
                        start_allocated = None
                    started = time.perf_counter()
                semantic, scalars = session.observe(context[start : start + session.chunk_size])
                if selected:
                    losses = []
                    for alpha in grid:
                        scenario_losses = []
                        for future in scenario["futures"]:
                            branch = session.clone()
                            branch.commit(alpha)
                            # Every branch sees the same remaining context and future continuation.
                            remaining = context[start + session.chunk_size :] + future.get("continuation_ids", [])
                            branch.consume(remaining, "baseline")
                            for query in future["queries"]:
                                scenario_losses.append(branch.score_answer(query["query_ids"], query["answer_ids"]))
                        losses.append(sum(scenario_losses) / len(scenario_losses))
                    if any(not math.isfinite(loss) for loss in losses):
                        raise ValueError(
                            f"nonfinite counterfactual loss for {scenario['id']} at boundary {boundary}; "
                            "check TTT checkpoint, update scale, and input scenario"
                        )
                    if on_cuda:
                        torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - started
                    gib = 1024 ** 3
                    peak_allocated = torch.cuda.max_memory_allocated(device) if on_cuda else None
                    peak_reserved = torch.cuda.max_memory_reserved(device) if on_cuda else None
                    minimum = min(losses)
                    best_index = next(i for i, loss in enumerate(losses) if loss <= minimum + tie_tolerance)
                    row = {
                        "id": scenario["id"], "group_id": scenario["group_id"], "split": scenario["split"],
                        "boundary": boundary, "state_policy": state_policy,
                        "semantic": semantic.squeeze(0).cpu().tolist(),
                        "scalars": scalars.squeeze(0).cpu().tolist(),
                        "alpha_star": grid[best_index], "grid": grid, "losses": losses,
                        "benefits": [losses[0] - loss for loss in losses],
                        "label_time_s": elapsed,
                        "start_allocated_gib": start_allocated / gib if on_cuda else None,
                        "peak_allocated_gib": peak_allocated / gib if on_cuda else None,
                        "peak_reserved_gib": peak_reserved / gib if on_cuda else None,
                        "extra_peak_allocated_gib": (peak_allocated - start_allocated) / gib if on_cuda else None,
                    }
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
                session.commit(_choose_alpha(session, state_policy, semantic, scalars))
    return count


def summarize_label_profile(path: str) -> dict:
    """Summarize per-label cost without loading model weights or feature tensors."""
    times = []
    allocated = []
    reserved = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            times.append(row["label_time_s"])
            if row["peak_allocated_gib"] is not None:
                allocated.append(row["peak_allocated_gib"])
                reserved.append(row["peak_reserved_gib"])
    if not times:
        raise ValueError("no labels to summarize")
    ordered = sorted(times)
    return {
        "labels": len(times),
        "mean_label_time_s": statistics.mean(times),
        "median_label_time_s": statistics.median(times),
        "p95_label_time_s": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max_label_time_s": ordered[-1],
        "max_peak_allocated_gib": max(allocated) if allocated else None,
        "max_peak_reserved_gib": max(reserved) if reserved else None,
    }


def _read_label_files(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    if not rows:
        raise ValueError("no controller labels were found")
    return rows


def train_controller(train_paths: list[str], dev_paths: list[str], output: str, layers: tuple[int, ...],
                     width: int = 128, semantic_size: int = 64, epochs: int = 20, batch_size: int = 64,
                     lr: float = 1e-3, seed: int = 42) -> dict:
    if not layers or tuple(sorted(set(layers))) != layers or any(layer < 0 for layer in layers):
        raise ValueError("layers must be sorted, unique, nonnegative TTT layer IDs")
    train_rows = _read_label_files(train_paths)
    dev_rows = _read_label_files(dev_paths)
    if any(row["split"] != "train" for row in train_rows) or any(row["split"] != "dev" for row in dev_rows):
        raise ValueError("train and dev label files must have their respective split markers")
    if {row["group_id"] for row in train_rows} & {row["group_id"] for row in dev_rows}:
        raise ValueError("train and dev label files share a source group")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(seed)
    random.seed(seed)
    hidden_size = len(train_rows[0]["semantic"])
    scalar_size = 4 + 3 * len(layers)
    for row in train_rows + dev_rows:
        if len(row["semantic"]) != hidden_size or len(row["scalars"]) != scalar_size:
            raise ValueError("controller label features have inconsistent dimensions")
        if not 0.0 <= row["alpha_star"] <= 1.0:
            raise ValueError("alpha_star is outside [0,1]")
    train_semantic = torch.tensor([row["semantic"] for row in train_rows], dtype=torch.float32)
    train_scalars = torch.tensor([row["scalars"] for row in train_rows], dtype=torch.float32)
    train_labels = torch.tensor([row["alpha_star"] for row in train_rows], dtype=torch.float32)
    dev_semantic = torch.tensor([row["semantic"] for row in dev_rows], dtype=torch.float32)
    dev_scalars = torch.tensor([row["scalars"] for row in dev_rows], dtype=torch.float32)
    dev_labels = torch.tensor([row["alpha_star"] for row in dev_rows], dtype=torch.float32)
    controller = ForgettingController(hidden_size, scalar_size, width, semantic_size)
    controller.set_statistics(train_scalars)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=lr)
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    history = []
    for epoch in range(epochs):
        controller.train()
        order = torch.randperm(len(train_rows))
        epoch_loss_sum = 0.0
        for indices in order.split(batch_size):
            optimizer.zero_grad()
            raw = controller(train_semantic[indices], train_scalars[indices])
            loss = F.mse_loss(raw, train_labels[indices])
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() * len(indices)
        controller.eval()
        with torch.no_grad():
            dev_loss = F.mse_loss(controller(dev_semantic, dev_scalars), dev_labels).item()
        if not math.isfinite(dev_loss):
            raise ValueError("nonfinite dev loss; inspect saved scalar features and learning rate")
        history.append({"epoch": epoch + 1, "train_mse": epoch_loss_sum / len(train_rows), "dev_mse": dev_loss})
        if dev_loss < best_loss:
            best_loss, best_epoch = dev_loss, epoch + 1
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in controller.state_dict().items()}
    checkpoint = {
        "state_dict": best_state, "layers": list(layers), "hidden_size": hidden_size,
        "scalar_size": scalar_size, "width": width, "semantic_size": semantic_size,
        "best_epoch": best_epoch, "dev_mse": best_loss,
    }
    torch.save(checkpoint, output)
    Path(output + ".metrics.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "train_states": len(train_rows), "dev_states": len(dev_rows),
        "best_epoch": best_epoch, "dev_mse": best_loss,
    }


def evaluate(model, scenarios: list[dict], output: str, policies: list[str], controller=None) -> dict:
    summary = {policy: [] for policy in policies}
    with open(output, "w", encoding="utf-8") as sink:
        for scenario in scenarios:
            for future_index, future in enumerate(scenario["futures"]):
                for policy in policies:
                    session = CBFSession(model, controller)
                    alphas = session.consume(scenario["context_ids"] + future.get("continuation_ids", []), policy)
                    losses = [session.score_answer(q["query_ids"], q["answer_ids"]) for q in future["queries"]]
                    mean_loss = sum(losses) / len(losses)
                    summary[policy].append(mean_loss)
                    sink.write(json.dumps({
                        "id": scenario["id"], "group_id": scenario["group_id"], "split": scenario["split"],
                        "future_index": future_index, "policy": policy, "query_losses": losses,
                        "mean_loss": mean_loss, "alphas": alphas,
                    }, ensure_ascii=False) + "\n")
    result = {}
    for policy, losses in summary.items():
        item = {"scenarios": len(losses), "mean_nll": sum(losses) / len(losses)}
        if policy != "baseline" and "baseline" in summary:
            gains = [base - current for base, current in zip(summary["baseline"], losses)]
            item["mean_gain_vs_baseline"] = sum(gains) / len(gains)
            item["harmful_forgetting_fraction"] = sum(gain < 0 for gain in gains) / len(gains)
        result[policy] = item
    Path(output + ".summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
