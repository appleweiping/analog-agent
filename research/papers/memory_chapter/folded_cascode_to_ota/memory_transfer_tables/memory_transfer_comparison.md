# Cross-Task Memory Transfer Comparison

Primary cross-task transfer comparison table.

| Mode | Feasible Hit Rate | Avg Sim Calls | Avg Step to Feasible | Avg Repeated Failures | Warm-Start Rate | Avg Advice Count | Avg Advice Consumed | Advice Consumption Rate | Governance Block Rate | Retrieval Precision | Negative Transfer Risk | Harmful Transfer Rate | Wrong-Family Retrieval Rate | Harmful Advice Application Rate | Cross-Task Failure After Retrieval | Governance Rejection Rate | Advice-Aligned Selection Rate | Retrieval-to-Success Conversion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_memory | 1.0 | 3.0 | 0.0 | 2.75 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| governed_transfer | 1.0 | 2.0 | 0.0 | 1.75 | 1.0 | 15.0 | 1.0 | 0.066667 | 0.0 | 0.4634 | 0.5366 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| no_governance | 1.0 | 2.0 | 0.0 | 1.75 | 1.0 | 15.0 | 1.0 | 0.066667 | 0.0 | 0.4634 | 0.5366 | 1.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| forced_transfer | 1.0 | 3.0 | 0.0 | 2.75 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.15 | 0.85 | 1.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |
