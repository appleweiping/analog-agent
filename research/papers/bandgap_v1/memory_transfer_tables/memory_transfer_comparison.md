# Cross-Task Memory Transfer Comparison

Primary cross-task transfer comparison table.

| Mode | Feasible Hit Rate | Avg Sim Calls | Avg Step to Feasible | Avg Repeated Failures | Warm-Start Rate | Retrieval Precision | Negative Transfer Risk | Harmful Transfer Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_memory | 0.0 | 1.0 | 0.0 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 |
| governed_transfer | 0.0 | 1.0 | 0.0 | 0.5 | 0.0 | 0.3966 | 0.6034 | 0.0 |
| forced_transfer | 0.0 | 1.0 | 0.0 | 0.5 | 1.0 | 0.15 | 0.85 | 0.5 |
