import math

import pytest

from prompt_methods.mctsv8_vpw.policy import compute_progress, get_stage_policy


def test_root_depth_starts_at_zero_progress_and_shallow_width():
    policy = get_stage_policy(depth=1, max_depth=8)

    assert policy.progress == 0.0
    assert policy.stage == "shallow"
    assert policy.normal_width == 5
    assert policy.reserve_width == 3


def test_depth_boundaries_select_middle_and_deep_widths():
    shallow = get_stage_policy(depth=3, max_depth=8)
    middle = get_stage_policy(depth=4, max_depth=8)
    deep = get_stage_policy(depth=7, max_depth=8)

    assert shallow.stage == "shallow"
    assert (shallow.normal_width, shallow.reserve_width) == (5, 3)
    assert middle.stage == "middle"
    assert (middle.normal_width, middle.reserve_width) == (5, 0)
    assert deep.stage == "deep"
    assert (deep.normal_width, deep.reserve_width) == (3, 0)


def test_terminal_policy_has_no_generation_width():
    policy = get_stage_policy(depth=2, max_depth=8, is_terminal=True)

    assert policy.progress == 1.0
    assert policy.stage == "terminal"
    assert (policy.normal_width, policy.reserve_width) == (0, 0)


def test_single_depth_tree_has_finite_deep_progress():
    progress = compute_progress(depth=1, max_depth=1)
    policy = get_stage_policy(depth=1, max_depth=1)

    assert math.isfinite(progress)
    assert progress == 1.0
    assert policy.stage == "deep"
    assert (policy.normal_width, policy.reserve_width) == (3, 0)


def test_progress_clamps_depths_outside_configured_range():
    assert compute_progress(depth=0, max_depth=8) == 0.0
    assert compute_progress(depth=99, max_depth=8) == 1.0


def test_nonpositive_max_depth_is_rejected_before_search():
    with pytest.raises(ValueError, match="max_depth must be positive"):
        get_stage_policy(depth=1, max_depth=0)
