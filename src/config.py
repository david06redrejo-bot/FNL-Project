"""Small configuration objects shared by scripts and notebooks."""

from dataclasses import dataclass


RANDOM_STATE = 42
EXPECTED_CATEGORIES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass(frozen=True)
class ExperimentConfig:
    """Minimal experiment metadata used by runnable model files."""

    model_id: str
    model_name: str
    random_state: int = RANDOM_STATE
    validation_size: float = 0.2
    target_column: str = "y_category"
    literal_column: str = "Literal"
    code_column: str = "Code"
    id_column: str = "id"

