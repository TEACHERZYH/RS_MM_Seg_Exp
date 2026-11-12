"""Identity checks shared by evaluation and aggregation; no data/model loading."""
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_completion(path: Path, config_hash: str, checkpoint_hash: str, epochs=None) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("Invalid training completion receipt schema")
    if record.get("status") != "pass" or record.get("termination_reason") != "max_epochs":
        raise ValueError("Training did not complete the fixed budget")
    completed, requested = record.get("completed_epoch"), record.get("requested_epochs")
    if not isinstance(requested, int) or requested < 1 or completed != requested:
        raise ValueError("Training epoch budget mismatch")
    if epochs is not None and requested != epochs:
        raise ValueError("Training/config epoch mismatch")
    for key, expected in (("config_sha256", config_hash), ("last_model_sha256", checkpoint_hash)):
        if str(record.get(key, "")).upper() != expected.upper():
            raise ValueError(f"Training receipt {key} mismatch")
    return record


def validate_checkpoint(checkpoint: dict, config: dict, checkpoint_path: Path, config_hash: str) -> dict:
    if checkpoint.get("config") != config:
        raise ValueError("Checkpoint embedded configuration differs from evaluation configuration")
    epochs = config["train"]["epochs"]
    if checkpoint.get("epoch") != epochs:
        raise ValueError("Checkpoint is not the fixed-budget final epoch")
    if checkpoint_path.name != "last_model.pt":
        raise ValueError("Fixed-budget evaluation requires last_model.pt")
    receipt_path = checkpoint_path.parent / "training_complete.json"
    validate_completion(receipt_path, config_hash, sha256_file(checkpoint_path), epochs)
    return {"status": "pass", "epoch": epochs, "training_complete_sha256": sha256_file(receipt_path)}
