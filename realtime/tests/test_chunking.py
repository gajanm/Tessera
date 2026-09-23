"""Sliding-window bookkeeping: how many chunks are ready and which frames each spans."""
import numpy as np
import pytest

from pipeline import count_available_chunks, plan_chunks


def reference_spans(buf_len, is_first, chunk_size, overlap, workers):
    """The inline logic process_chunks_parallel used before it was extracted."""
    stride = chunk_size - overlap
    if stride <= 0:
        return []
    if is_first:
        if buf_len < chunk_size:
            return []
        max_chunks = 1 + (buf_len - chunk_size) // stride
    else:
        if buf_len < stride:
            return []
        max_chunks = buf_len // stride

    spans = []
    for ci in range(min(max_chunks, workers)):
        if is_first:
            start, end = ci * stride, ci * stride + chunk_size
            length = len(range(*slice(start, end).indices(buf_len)))
        elif ci == 0:
            start, end = -overlap, stride
            length = overlap + len(range(*slice(0, stride).indices(buf_len)))
        else:
            start, end = ci * stride - overlap, ci * stride - overlap + chunk_size
            length = len(range(*slice(start, end).indices(buf_len)))
        if length < chunk_size:
            break
        spans.append((start, end))
    return spans


# ── count_available_chunks ───────────────────────────────────────

def test_first_chunk_needs_a_full_window():
    assert count_available_chunks(15, True, 16, 6) == 0
    assert count_available_chunks(16, True, 16, 6) == 1
    assert count_available_chunks(25, True, 16, 6) == 1
    assert count_available_chunks(26, True, 16, 6) == 2


def test_continuation_needs_only_a_stride():
    assert count_available_chunks(9, False, 16, 6) == 0
    assert count_available_chunks(10, False, 16, 6) == 1
    assert count_available_chunks(35, False, 16, 6) == 3


def test_non_positive_stride_yields_nothing():
    assert count_available_chunks(100, True, 16, 16) == 0
    assert count_available_chunks(100, False, 16, 20) == 0
    assert plan_chunks(100, False, 16, 20, 4) == []


# ── plan_chunks ──────────────────────────────────────────────────

def test_consecutive_chunks_share_exactly_overlap_frames():
    """The whole stitching design rests on this."""
    for is_first in (True, False):
        spans = plan_chunks(200, is_first, 16, 6, 8)
        assert len(spans) > 1
        for (s0, e0), (s1, e1) in zip(spans, spans[1:]):
            assert e0 - s1 == 6
            assert e0 - s0 == 16 and e1 - s1 == 16


def test_continuation_first_chunk_reaches_back_into_the_previous_tail():
    spans = plan_chunks(100, False, 16, 6, 4)
    assert spans[0] == (-6, 10)
    assert all(s >= 0 for s, _ in spans[1:])


def test_first_pass_starts_at_zero():
    assert plan_chunks(100, True, 16, 6, 4)[0] == (0, 16)


def test_never_reads_past_the_buffer():
    for buf_len in range(0, 80):
        for is_first in (True, False):
            for s, e in plan_chunks(buf_len, is_first, 16, 6, 8):
                assert e <= buf_len


def test_capped_by_worker_count():
    assert len(plan_chunks(500, True, 16, 6, 3)) == 3
    assert len(plan_chunks(500, False, 16, 6, 1)) == 1
    assert plan_chunks(500, True, 16, 6, 0) == []


@pytest.mark.parametrize("chunk_size,overlap", [(16, 6), (8, 2), (20, 10), (4, 2)])
def test_matches_the_original_inline_logic(chunk_size, overlap):
    for buf_len in range(0, 120):
        for is_first in (True, False):
            for workers in (1, 2, 4, 8):
                assert plan_chunks(buf_len, is_first, chunk_size, overlap, workers) \
                    == reference_spans(buf_len, is_first, chunk_size, overlap, workers)


# ── buffering ────────────────────────────────────────────────────

def test_add_frame_reports_readiness_and_counts_down(pipe):
    frame = np.zeros((48, 64, 3), np.uint8)
    pipe.configure(chunk_size=8, overlap=2)

    assert pipe.frames_until_ready() == 8
    for i in range(7):
        assert pipe.add_frame(frame) is False
        assert pipe.frames_until_ready() == 7 - i
    assert pipe.add_frame(frame) is True
    assert pipe.frames_until_ready() == 0


def test_after_the_first_chunk_only_a_stride_is_needed(pipe):
    pipe.configure(chunk_size=8, overlap=2)
    pipe.is_first_chunk = False
    assert pipe.frames_until_ready() == 6


def test_configure_clamps_out_of_range_values(pipe):
    pipe.configure(chunk_size=999, overlap=999, conf_thre=999)
    assert pipe.chunk_size == 20
    assert pipe.overlap == pipe.chunk_size // 2
    assert pipe.conf_thre == 0.5

    pipe.configure(chunk_size=1, overlap=0, conf_thre=0)
    assert pipe.chunk_size == 4
    assert pipe.overlap == 2
    assert pipe.conf_thre == 0.01
