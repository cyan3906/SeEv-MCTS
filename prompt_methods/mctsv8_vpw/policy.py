from dataclasses import dataclass


@dataclass(frozen=True)
class StagePolicy:
    progress: float
    stage: str
    normal_width: int
    reserve_width: int


def compute_progress(depth: int, max_depth: int, is_terminal: bool = False) -> float:
    if max_depth < 1:
        raise ValueError("max_depth must be positive")
    if is_terminal:
        return 1.0
    if max_depth <= 1:
        return 1.0
    progress = (depth - 1) / (max_depth - 1)
    return max(0.0, min(1.0, float(progress)))


def get_stage_policy(
    depth: int,
    max_depth: int,
    is_terminal: bool = False,
) -> StagePolicy:
    progress = compute_progress(depth, max_depth, is_terminal=is_terminal)
    if is_terminal:
        return StagePolicy(progress, "terminal", 0, 0)
    if progress < 0.375:
        return StagePolicy(progress, "shallow", 5, 3)
    if progress < 0.75:
        return StagePolicy(progress, "middle", 5, 0)
    return StagePolicy(progress, "deep", 3, 0)
