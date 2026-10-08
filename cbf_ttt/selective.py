"""Optional retention of direct update cohorts; the original runtime is unchanged."""

import copy
import itertools

import torch

from cbf_ttt.runtime import CBFSession


GRID = tuple(itertools.product((0.0, 0.5, 1.0), repeat=3))
FIXED = {
    "global_clear": (0.0, 0.0, 0.0),
    "global_half": (0.5, 0.5, 0.5),
    "window2": (0.0, 0.0, 1.0),
    "window3": (0.0, 1.0, 1.0),
}


class CohortSession(CBFSession):
    """R1/R2 track retained direct contributions, not removable semantic facts.

    Old history is the residual M-R1-R2, including model-dtype rounding error.
    Removing a cohort does not undo its earlier effects on KV or later deltas.
    """

    def __init__(self, model, cache=None, recent1=None, recent2=None):
        super().__init__(model, cache=cache)
        self.recent1 = recent1 or {}
        self.recent2 = recent2 or {}

    def clone(self):
        return CohortSession(self.model, copy.deepcopy(self.cache),
                             copy.deepcopy(self.recent1), copy.deepcopy(self.recent2))

    def commit(self, coefficient):
        self.commit_cohorts((1.0 - coefficient,) * 3)

    def commit_both(self, alpha, write_gate):
        if write_gate != 1.0:
            raise ValueError("selective forgetting fixes write_gate=1")
        self.commit_cohorts((alpha,) * 3)

    @torch.inference_mode()
    def commit_cohorts(self, retention):
        if len(retention) != 3 or any(not 0 <= a <= 1 for a in retention):
            raise ValueError("three retention rates in [0, 1] are required")
        if set(self.cache.cbf_candidates) != set(self.layers):
            raise RuntimeError("a complete pending chunk is required before commit")
        ah, a2, a1 = retention
        # Independent float32 copies keep provenance when native memory is BF16.
        next1 = {i: self.cache.cbf_candidates[i].float().clone() for i in self.layers}
        next2 = {i: a1 * self.recent1[i] for i in self.recent1}
        if ah == a2 == a1:
            # Preserve the exact numerical path of the original global update.
            CBFSession.commit_both(self, ah, 1.0)
        else:
            for i in self.layers:
                delta = self.cache.cbf_candidates[i]
                old = self.cache.cbf_memory.get(i)
                if old is None:
                    result = delta.clone()
                else:
                    r1 = self.recent1.get(i, 0.0)
                    r2 = self.recent2.get(i, 0.0)
                    history = old.float() - r1 - r2
                    result = (ah * history + a2 * r2 + a1 * r1 + next1[i]).to(delta.dtype)
                self.cache.cbf_memory[i] = result
            self.cache.cbf_candidates = {}
            self.cache.cbf_collect = False
        self.recent1, self.recent2 = next1, next2

    def score_answer(self, query_ids, answer_ids):
        # Scoring needs a cloned KV/memory cache, not extra cohort buffer copies.
        reader = CBFSession(self.model, cache=self.cache)
        return reader.score_answer(query_ids, answer_ids)

    def advance(self, ids, retention=(1.0, 1.0, 1.0)):
        if len(ids) != self.chunk_size:
            raise ValueError("advance requires one complete TTT chunk")
        self._forward(ids, collect=True)
        self.commit_cohorts(retention)
