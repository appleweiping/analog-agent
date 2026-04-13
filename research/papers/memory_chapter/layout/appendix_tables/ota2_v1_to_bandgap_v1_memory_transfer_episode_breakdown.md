# Cross-Task Memory Transfer Breakdown

Per-episode cross-task transfer breakdown for appendix and failure analysis.

| Mode | Episode | Warm Start | Real Sim Calls | Step to Feasible | Repeated Failures | Retrieved Episodes | Advice Count | Advice Consumed | Governance Blocks | Negative Transfer Risk | Harmful Transfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_memory | 0 | False | 1 | na | 0 | 0 | 0 | 0 | 0 | 0.0 | False |
| no_memory | 1 | False | 1 | na | 1 | 0 | 0 | 0 | 0 | 0.0 | False |
| no_memory | 2 | False | 1 | na | 1 | 0 | 0 | 0 | 0 | 0.0 | False |
| no_memory | 3 | False | 1 | na | 1 | 0 | 0 | 0 | 0 | 0.0 | False |
| governed_transfer | 0 | False | 1 | na | 0 | 4 | 15 | 0 | 1 | 0.6034 | False |
| governed_transfer | 1 | False | 1 | na | 1 | 4 | 15 | 0 | 1 | 0.6034 | False |
| governed_transfer | 2 | False | 1 | na | 1 | 4 | 15 | 0 | 1 | 0.6034 | False |
| governed_transfer | 3 | False | 1 | na | 1 | 4 | 15 | 0 | 1 | 0.6034 | False |
| no_governance | 0 | True | 1 | na | 0 | 4 | 15 | 1 | 0 | 0.6034 | False |
| no_governance | 1 | True | 1 | na | 1 | 4 | 15 | 1 | 0 | 0.6034 | True |
| no_governance | 2 | True | 1 | na | 1 | 4 | 15 | 1 | 0 | 0.6034 | True |
| no_governance | 3 | True | 1 | na | 1 | 4 | 15 | 1 | 0 | 0.6034 | True |
| forced_transfer | 0 | True | 1 | na | 0 | 1 | 1 | 1 | 0 | 0.85 | False |
| forced_transfer | 1 | True | 1 | na | 1 | 1 | 1 | 1 | 0 | 0.85 | True |
| forced_transfer | 2 | True | 1 | na | 1 | 1 | 1 | 1 | 0 | 0.85 | True |
| forced_transfer | 3 | True | 1 | na | 1 | 1 | 1 | 1 | 0 | 0.85 | True |
