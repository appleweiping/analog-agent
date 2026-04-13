# Memory Negative-Transfer Case Study: ota2-v1 -> ldo-v1

ota2-v1 -> ldo-v1 shows governed cross-family reuse holding harmful transfer at 0.000, while no-governance rises to 1.000 and forced transfer rises to 1.000.

## Structured Metrics

- Governed harmful-transfer rate: `0.0`
- No-governance harmful-transfer rate: `1.0`
- Forced-transfer harmful-transfer rate: `1.0`
- Governed avg real simulation calls: `3.0`
- No-governance avg real simulation calls: `2.0`
- Forced avg real simulation calls: `3.0`
- Governance block rate: `1.0`
- Avg negative-transfer risk: `0.5934`

## Interpretation

- This pair is suitable for the paper's negative-transfer discussion because the governed path keeps harmful reuse low while the relaxed paths reveal measurable risk.
- It can be cited as evidence that the memory layer is not only helpful under same-family reuse, but also actively guarded under cross-family mismatch.