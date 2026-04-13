# Repeated-Episode Memory Ablation Comparison

Primary repeated-episode memory ablation comparison table.

| Mode | Feasible Hit Rate | Avg Sim Calls | Avg Step to Feasible | Avg Repeated Failures | Warm-Start Rate | Avg Advice Count | Avg Advice Consumed | Advice Consumption Rate | Governance Block Rate | Retrieval Precision | Negative Transfer Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_memory | 0.0 | 1.0 | 0.0 | 0.666667 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| episodic_retrieval_only | 0.666667 | 1.0 | 0.0 | 0.666667 | 0.666667 | 7.0 | 0.0 | 0.0 | 0.0 | 0.6 | 0.066667 |
| episodic_plus_reflection | 0.666667 | 1.0 | 0.0 | 0.666667 | 0.666667 | 7.0 | 0.666667 | 0.12963 | 0.0 | 0.6 | 0.066667 |
| full_memory | 0.666667 | 1.0 | 0.0 | 0.666667 | 0.666667 | 7.0 | 1.333333 | 0.259259 | 0.0 | 0.6 | 0.066667 |
