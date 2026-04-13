# Cross-Task Memory Transfer Breakdown

Per-episode cross-task transfer breakdown for appendix and failure analysis.

| Mode | Episode | Warm Start | Real Sim Calls | Step to Feasible | Repeated Failures | Retrieved Episodes | Negative Transfer Risk | Harmful Transfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_memory | 0 | False | 2 | 0 | 1 | 0 | 0.0 | False |
| no_memory | 1 | False | 2 | 0 | 2 | 0 | 0.0 | False |
| governed_transfer | 0 | True | 1 | 0 | 0 | 2 | 0.5366 | False |
| governed_transfer | 1 | True | 1 | 0 | 1 | 2 | 0.5366 | False |
| forced_transfer | 0 | True | 2 | 0 | 1 | 1 | 0.85 | True |
| forced_transfer | 1 | True | 2 | 0 | 2 | 1 | 0.85 | True |
