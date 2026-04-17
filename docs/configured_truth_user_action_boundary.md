# Configured-Truth User Action Boundary

This repository now formalizes the boundary between what the codebase can guarantee and what only a user can supply.

## Repo-Managed Guarantees

- `paper_mode` blocks mock truth from paper-facing runs.
- configured-truth requests are validated against a formal contract.
- native artifact runs persist replay-oriented manifests.
- native artifact rerunability can be checked structurally.
- container runtime contract and open-PDK readiness can be reviewed without modifying code.

## User-Managed Inputs

The following inputs are intentionally not created or downloaded automatically by the repository:

- a structured external PDK root such as `sky130A`
- an external model card when stronger configured truth is desired
- local mount paths and storage locations for those assets
- Docker installation/exposure for first real container smoke tests

## Practical Rule

If the repository reports:

- `demonstrator_only`
- `configured_truth_not_ready`
- or `configured_truth_candidate_ready`

then stronger configured-truth claims must remain withheld until the user-managed assets are actually present and validated.
