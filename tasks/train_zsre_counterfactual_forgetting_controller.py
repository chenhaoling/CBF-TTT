"""Run the residual forgetting controller on full ZsRE counterfactual sequences."""

import argparse
import json

from tasks.build_zsre_counterfactual_forgetting import PROTOCOL as DATA_PROTOCOL
from tasks.train_zsre_forgetting_controller_v2 import run


PROTOCOL = "zsre_counterfactual_forgetting_controller_loso_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--scope-root", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True); parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128); args = parser.parse_args()
    result = run(args, protocol=PROTOCOL, data_protocol=DATA_PROTOCOL,
                 output_stem="counterfactual_gate",
                 checkpoint_filename="counterfactual_forgetting_gate.pt",
                 comparison_seed=9001)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
