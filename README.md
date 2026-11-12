# Anonymous QALF Reproduction Repository

This repository contains the code and frozen configurations for a blinded reproduction of a lightweight optical-DSM semantic-segmentation system under missing and degraded inputs.

## Contents

- `src/`, `train.py`, `evaluate.py`: model, dataset loader, losses, training and evaluation.
- `configs/minimal_claim_m2_r2_20261102_fair/`: the current fair matrix. Its fixed-late route uses availability-masked averaging when DSM is absent.
- `scripts/`: data preparation, holdout construction, protocol evaluation, M2 severity evaluation, provenance checks, efficiency/diagnostic exports, command planning and tests.
- `requirements.txt`: Python dependencies.

The repository does not redistribute benchmark data, prepared tiles, checkpoints, trained weights, numerical result logs, or internal experiment records. Results must be regenerated under the frozen matrix before they are used as evidence.

## Quick checks

```bash
python -m compileall -q src scripts train.py evaluate.py
python scripts/test_reproduction_plan.py
python scripts/test_seed_statistics.py
python scripts/test_minimal_claim_repro_contracts.py
```

Generate a validated command plan without running experiments:

```bash
python scripts/plan_reproduction.py --output outputs/reproduction_commands.json
```

The plan checks configuration hashes, training-stage dependencies, fixed budgets, checkpoint endpoints, split hashes, evaluation identities, scenario counts and trial policy. Holdout commands are listed separately.

For data preparation and evaluation details, see the files under `configs/minimal_claim_m2_r2_20261102_fair/` and `scripts/`.

