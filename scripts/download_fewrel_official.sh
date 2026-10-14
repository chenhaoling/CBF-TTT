#!/usr/bin/env bash
set -euo pipefail

DEST=${DEST:-/home/ctj/data/fewrel_official_20261015}
COMMIT=278a2315d2138810a379cd8d5718914dc56e2582

test ! -e "$DEST"
git clone --depth 1 https://github.com/thunlp/FewRel.git "$DEST"
test "$(git -C "$DEST" rev-parse HEAD)" = "$COMMIT"
cd "$DEST"
echo '8e9e58a05d559be773cbe18a8d4a16ba17e6d397663f9f05a056f894a94b92c1  data/train_wiki.json' | sha256sum -c -
echo '76bb44c2485b6faed358a3ce040eadafcc8a8835245cd7e6afc1bef672d4a0bf  data/val_wiki.json' | sha256sum -c -
echo 'eeedd25512a8e4c2bc06b78009836d27de81807bcba6ed929debd5cd713c07ea  data/pid2name.json' | sha256sum -c -
