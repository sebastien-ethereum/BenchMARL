from dataclasses import dataclass, MISSING

@dataclass
class TaskConfig:
    n_bidders: int = MISSING
    max_steps: int = MISSING
    min_value: float = MISSING
    max_value: float = MISSING
