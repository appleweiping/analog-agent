# Repeated-Episode Memory Ablation Comparison

Primary repeated-episode memory ablation comparison table.

| Mode | Feasible Hit Rate | Avg Sim Calls | Avg Step to Feasible | Avg Repeated Failures | Warm-Start Rate | Avg Advice Count | Avg Advice Consumed | Advice Consumption Rate | Governance Block Rate | Retrieval Precision | Negative Transfer Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_memory | 1.0 | 3.0 | 0.0 | 2.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| episodic_retrieval_only | 1.0 | 2.1 | 0.0 | 2.0 | 0.9 | 22.8 | 0.0 | 0.0 | 0.0 | 0.81 | 0.09 |
| episodic_plus_reflection | 1.0 | 2.1 | 0.0 | 2.0 | 0.9 | 22.8 | 0.9 | 0.045966 | 0.0 | 0.81 | 0.09 |
| full_memory | 1.0 | 2.1 | 0.0 | 1.7 | 0.9 | 22.8 | 1.8 | 0.091931 | 0.0 | 0.81 | 0.09 |
