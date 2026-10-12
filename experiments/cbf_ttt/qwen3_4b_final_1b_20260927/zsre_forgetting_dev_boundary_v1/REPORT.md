# ZsRE forgetting sparse boundary selection

The frozen controller was replayed at 0.625, 0.650, 0.675, 0.725, 0.750, and 0.775, selected from the adjacent coarse interval before this run. The independent audit passed and confirmed that test remained sealed.

Thresholds 0.625 and 0.675 passed all original checks. The registered reliability ordering selected 0.675:

- query accuracy: 77.91%;
- action accuracy: 96.74%;
- primary initial / hard retain / primary update: 100.00 / 98.05 / 92.19%;
- query gain over cosine-local: +6.27 points, 95% CI `[4.70, 7.95]`;
- action gain: +12.37 points, CI `[8.98, 15.89]`.

Runtime was 68.76 seconds; peak allocated/reserved CUDA memory was 8.233/11.297 GiB. The calibration-trained head and threshold 0.675 were then frozen for the one-time held-out test.
