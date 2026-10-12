# ZsRE forgetting threshold diagnostic

The frozen calibration controller was causally replayed on development at thresholds 0.1 through 0.9. Every threshold received its own evolving memory state and query outcomes. The independent audit recomputed all 27 controller traces and verified that test remained sealed.

No coarse point passed all six checks. The main boundary was:

| Threshold | Query | Action | Hard retain | Update |
|---:|---:|---:|---:|---:|
| 0.6 | 77.18% | 95.05% | 94.92% | 90.23% |
| 0.7 | 77.85% | 96.61% | 98.05% | 91.80% |
| 0.8 | 77.88% | 97.01% | 100.00% | 91.02% |

Threshold 0.7 missed the 92% update floor by one of 256 events. This justified the preregistered sparse boundary sweep without changing the controller or opening test. Runtime was 86.83 seconds; peak allocated/reserved CUDA memory was 8.233/11.297 GiB.
