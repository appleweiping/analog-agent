# Cross-Task Memory Transfer Summary

Chapter-level transfer summary across same-family and cross-family settings.

| Transfer Pair | Transfer Kind | Governed Avg Sim Calls | No-Governance Harmful Rate | Forced Harmful Rate | Governance Blocks Harm | Governed Beneficial |
| --- | --- | --- | --- | --- | --- | --- |
| ota2-v1->folded_cascode-v1 | same_family | 2.0 | 1.0 | 1.0 | True | True |
| folded_cascode-v1->ota2-v1 | same_family | 2.0 | 1.0 | 1.0 | True | True |
| ota2-v1->ldo-v1 | cross_family | 3.0 | 1.0 | 1.0 | True | False |
| ota2-v1->bandgap-v1 | cross_family | 1.0 | 0.75 | 0.75 | True | False |
| folded_cascode-v1->ldo-v1 | cross_family | 3.0 | 1.0 | 1.0 | True | False |
| folded_cascode-v1->bandgap-v1 | cross_family | 1.0 | 0.5 | 0.5 | True | False |
