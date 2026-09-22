"""The boundary contract consumers depend on: a gapless partition of the asset.

These are pure-function tests — no artifact, no GPU, no video.
"""
import pytest

from lib.model.ai_shot_boundary_model import (
    SCHEMA_VERSION,
    assemble_result,
    build_partition,
    merge_ranges,
    prune_non_context_ranges,
)

ELEMENT_KEYS = {
    "start_frame", "end_frame", "start_seconds", "end_seconds",
    "shot_type", "transition_in", "filled",
}


def assert_partition(boundaries, frame_count, duration):
    assert boundaries, "expected at least one boundary"
    assert boundaries[0]["start_frame"] == 0
    assert boundaries[0]["start_seconds"] == 0.0
    assert boundaries[0]["transition_in"] is None
    for index in range(1, len(boundaries)):
        previous, current = boundaries[index - 1], boundaries[index]
        assert current["start_frame"] == previous["end_frame"], f"frame gap or overlap at {index}"
        assert current["start_seconds"] == previous["end_seconds"], f"time gap or overlap at {index}"
    assert boundaries[-1]["end_frame"] == frame_count
    assert boundaries[-1]["end_seconds"] == round(duration, 3)
    for boundary in boundaries:
        assert set(boundary) == ELEMENT_KEYS
        assert boundary["end_frame"] > boundary["start_frame"]
        assert boundary["end_seconds"] > boundary["start_seconds"]
        for key in ("start_seconds", "end_seconds"):
            assert round(boundary[key], 3) == boundary[key], f"{key} is not ms-rounded"
        if boundary["filled"]:
            assert boundary["shot_type"] is None and boundary["transition_in"] is None


def partition(ranges, frame_count, duration, fps=30.0, shot_types=None, transitions=None):
    return build_partition(
        ranges=ranges,
        shot_types=shot_types if shot_types is not None else ["General"] * len(ranges),
        transitions=transitions if transitions is not None else ["Hard_Cut"] * len(ranges),
        frame_count=frame_count,
        fps=fps,
        duration=duration,
    )


@pytest.mark.parametrize("ranges,frame_count,duration", [
    ([[0, 30], [30, 60]], 60, 2.0),              # contiguous
    ([[0, 60]], 60, 2.0),                        # single shot covering everything
    ([[0, 40], [30, 60]], 60, 2.0),              # overlapping -> trimmed
    ([[0, 20], [40, 59]], 60, 2.0),              # gap and short tail -> filled
    ([[0, 30], [30, 600]], 60, 2.0),             # past the last frame -> clamped
    ([], 60, 2.0),                               # no predictions at all
    ([[0, 10]], 15, 0.5),                        # short asset
    ([[0, 30], [30, 60]], 60, 1.9),              # duration shorter than the frames
    ([[0, 30], [30, 60]], 60, 0.9),              # ...so short a shot collapses
    ([[0, 30], [30, 60]], 60, 2.5),              # duration longer than the frames
])
def test_partition_invariants(ranges, frame_count, duration):
    boundaries, _ = partition(ranges, frame_count, duration)
    assert_partition(boundaries, frame_count, duration)


@pytest.mark.parametrize("frame_count,duration,fps", [(10, 0.0, 30.0), (0, 1.0, 30.0), (10, 1.0, 0.0)])
def test_degenerate_assets_yield_no_boundaries(frame_count, duration, fps):
    boundaries, counts = partition([[0, 10]], frame_count, duration, fps=fps)
    assert boundaries == []
    assert counts == {"shot_type": {}, "transition_in": {}}


def test_ranges_are_half_open():
    # A shot [0, 30) at 30fps covers 0.0-1.0s.
    boundaries, _ = partition([[0, 30]], 30, 1.0)
    assert [(b["start_frame"], b["end_frame"], b["end_seconds"]) for b in boundaries] == [(0, 30, 1.0)]


def test_adjacent_shots_produce_one_boundary_each():
    """Half-open ranges must not overlap, or every cut becomes two boundaries
    separated by a one-frame sliver."""
    boundaries, _ = partition([[0, 120], [120, 240], [240, 360]], 360, 12.0)
    assert [(b["start_frame"], b["end_frame"]) for b in boundaries] == [(0, 120), (120, 240), (240, 360)]
    assert [(b["start_seconds"], b["end_seconds"]) for b in boundaries] == [
        (0.0, 4.0), (4.0, 8.0), (8.0, 12.0)
    ]


def test_transition_in_is_each_shots_own_inter_label():
    """The inter label describes how a shot is entered from the one before it."""
    boundaries, _ = partition(
        [[0, 30], [30, 60], [60, 90]], 90, 3.0,
        transitions=["New_Start", "Hard_Cut", "Sudden_Jump"],
    )
    assert [b["transition_in"] for b in boundaries] == [None, "Hard_Cut", "Sudden_Jump"]


def test_transition_shots_stay_labelled_in_the_partition():
    ranges = [[0, 30], [30, 40], [40, 90]]
    result = assemble_result(
        model="m", model_version=1.0, mode="clean_shot", fps=30.0, duration=3.0,
        frame_count=90, window_frames=100, context_frames=0, decode_backend="ffmpeg_cpu",
        ranges=ranges,
        shot_types=["General", "Dissolve", "General"],
        transitions=["New_Start", "Transition_Source", "Transition"],
        in_shots=[True, False, True],
        timings={},
    )
    boundaries = result["boundaries"]
    assert_partition(boundaries, 90, 3.0)
    assert [(b["start_frame"], b["shot_type"], b["transition_in"]) for b in boundaries] == [
        (0, "General", None), (30, "Dissolve", "Transition_Source"), (40, "General", "Transition"),
    ]
    # clean_shot filters only the model's own shot list.
    assert [shot["start_frame"] for shot in result["shots"]] == [0, 40]


def test_uncovered_frames_are_filled_and_unlabelled():
    boundaries, _ = partition(
        [[0, 20], [40, 50]], 60, 2.0,
        shot_types=["General", "Dissolve"], transitions=["New_Start", "Hard_Cut"],
    )
    assert [(b["start_frame"], b["end_frame"], b["filled"]) for b in boundaries] == [
        (0, 20, False), (20, 40, True), (40, 50, False), (50, 60, True),
    ]
    # The shot after a gap keeps the label the model gave it.
    assert boundaries[2]["shot_type"] == "Dissolve"
    assert boundaries[2]["transition_in"] == "Hard_Cut"


def test_overlap_is_trimmed_off_the_later_shot():
    boundaries, _ = partition([[0, 40], [30, 60]], 60, 2.0)
    assert [(b["start_frame"], b["end_frame"]) for b in boundaries] == [(0, 40), (40, 60)]


def test_shots_past_the_last_frame_are_clamped():
    boundaries, _ = partition([[0, 30], [30, 100], [100, 130]], 60, 2.0)
    assert [(b["start_frame"], b["end_frame"]) for b in boundaries] == [(0, 30), (30, 60)]


def test_sub_millisecond_element_is_absorbed():
    # At 2400fps a one-frame shot is 0.4ms long and rounds away.
    boundaries, _ = partition(
        [[0, 1], [1, 1200], [1200, 1201], [1201, 2400]], 2400, 1.0, fps=2400.0,
        shot_types=["A", "B", "C", "D"], transitions=["New_Start", "Hard_Cut", "Hard_Cut", "Sudden_Jump"],
    )
    assert_partition(boundaries, 2400, 1.0)
    # The leading sliver joins its successor; the one mid-stream joins its predecessor.
    assert [(b["start_frame"], b["end_frame"], b["shot_type"]) for b in boundaries] == [
        (0, 1201, "B"), (1201, 2400, "D"),
    ]
    assert boundaries[0]["transition_in"] is None


def test_duration_shorter_than_frames_collapses_the_tail():
    boundaries, _ = partition([[0, 30], [30, 60]], 60, 0.9)
    assert [(b["start_frame"], b["end_frame"], b["end_seconds"]) for b in boundaries] == [(0, 60, 0.9)]


def test_label_counts_cover_the_partition():
    _, counts = partition(
        [[0, 10], [10, 20]], 30, 1.0,
        shot_types=["General", "Dissolve"], transitions=["New_Start", "Transition"],
    )
    assert counts == {"shot_type": {"General": 1, "Dissolve": 1}, "transition_in": {"Transition": 1}}


def test_result_shape():
    result = assemble_result(
        model="omnishotcut", model_version=1.0, mode="default", fps=25.0, duration=4.0,
        frame_count=100, window_frames=100, context_frames=0, decode_backend="ffmpeg_cpu",
        ranges=[[0, 40], [40, 100]], shot_types=["General", "General"],
        transitions=["New_Start", "Hard_Cut"], in_shots=[True, True],
        timings={"decode_wait_seconds": 0.1, "inference_seconds": 0.2, "analyze_seconds": 0.3},
    )
    assert result["schema_version"] == SCHEMA_VERSION == 2
    assert result["model_version"] == "1.0"
    assert result["duration_seconds"] == 4.0
    assert result["source_frame_count"] == 100
    assert result["duration_mismatch_seconds"] == 0.0
    assert result["analyze_seconds"] == 0.3
    assert "transition_after" not in str(result)
    assert result["shots"] == [
        {"start_frame": 0, "end_frame": 40, "intra": "General", "inter": "New_Start"},
        {"start_frame": 40, "end_frame": 100, "intra": "General", "inter": "Hard_Cut"},
    ]


# ── windowing helpers ────────────────────────────────────────────────────

def test_prune_is_identity_without_context():
    ranges = [[0, 10], [10, 40]]
    got = prune_non_context_ranges(ranges, ["A", "B"], ["X", "Y"], 100, 0)
    assert got == (ranges, ["A", "B"], ["X", "Y"])


def test_prune_drops_and_rebases_context_regions():
    # window 100, context 10 -> the usable region is local [10, 90)
    ranges = [[0, 5], [5, 50], [95, 100]]
    pruned, intra, inter = prune_non_context_ranges(
        ranges, ["A", "B", "C"], ["X", "Y", "Z"], 100, 10
    )
    assert pruned == [[0, 40]]        # [0,5] dropped (inside leading context), [5,50] rebased
    assert intra == ["B"] and inter == ["Y"]


def test_merge_places_each_window_at_its_own_offset():
    full_r, full_i, full_e = [], [], []
    merge_ranges(full_r, full_i, full_e, [[0, 50], [50, 100]], [0, 0], [1, 1], 0, new_start_inter_index=0)
    merge_ranges(full_r, full_i, full_e, [[0, 30]], [0], [1], 100, new_start_inter_index=0)
    assert full_r == [[0, 50], [50, 100], [100, 130]]


def test_merge_leaves_a_gap_after_a_window_that_stopped_short():
    """The next window starts at its own first frame, not where the previous
    window's shots ran out, so nothing after a short window shifts."""
    full_r, full_i, full_e = [], [], []
    merge_ranges(full_r, full_i, full_e, [[0, 90]], [0], [1], 0, new_start_inter_index=0)
    merge_ranges(full_r, full_i, full_e, [[0, 50], [50, 100]], [0, 0], [1, 1], 100, new_start_inter_index=0)
    assert full_r == [[0, 90], [100, 150], [150, 200]]
    boundaries, _ = partition(full_r, 200, 8.0, fps=25.0)
    assert [(b["start_frame"], b["end_frame"], b["filled"]) for b in boundaries] == [
        (0, 90, False), (90, 100, True), (100, 150, False), (150, 200, False),
    ]


def test_merge_joins_a_shot_continuing_across_the_seam():
    # Same intra label and a 'new_start' inter label means the next window's
    # first shot is a continuation, not a new shot.
    full_r, full_i, full_e = [[0, 100]], [7], [1]
    merge_ranges(full_r, full_i, full_e, [[0, 30], [30, 60]], [7, 7], [0, 1], 100, new_start_inter_index=0)
    assert full_r[0] == [0, 130]
    assert len(full_r) == 2


def test_merge_does_not_join_across_different_intra_labels():
    full_r, full_i, full_e = [[0, 100]], [8], [1]
    merge_ranges(full_r, full_i, full_e, [[0, 30]], [7], [0], 100, new_start_inter_index=0)
    assert full_r == [[0, 100], [100, 130]]
