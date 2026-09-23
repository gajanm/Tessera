"""The binary point-cloud wire format, decoded the way viewer.ts decodes it."""
import struct

import numpy as np
import pytest

from server import encode_point_cloud_binary_fast

HEADER = 16


def decode(buf):
    """Mirror of parseBinaryPointCloud in viewer.ts."""
    chunk_id, total, n, elapsed = struct.unpack_from('<iiif', buf, 0)
    pts = np.frombuffer(buf, np.float32, count=n * 3, offset=HEADER)
    cols = np.frombuffer(buf, np.uint8, count=n * 3, offset=HEADER + n * 12)
    return chunk_id, total, n, elapsed, pts.reshape(n, 3), cols.reshape(n, 3)


def test_round_trips_points_and_colors():
    pts = np.array([[1.5, -2.0, 3.25], [0, 0, 0], [1e3, -1e3, 0.5]], np.float32)
    cols = np.array([[255, 0, 10], [1, 2, 3], [0, 128, 255]], np.uint8)

    chunk_id, total, n, elapsed, got_pts, got_cols = decode(
        encode_point_cloud_binary_fast(pts, cols, 42, 9999, 1.25))

    assert (chunk_id, total, n) == (42, 9999, 3)
    assert elapsed == pytest.approx(1.25)
    assert np.array_equal(got_pts, pts)
    assert np.array_equal(got_cols, cols)


def test_payload_size_is_exactly_header_plus_15_bytes_per_point():
    n = 50
    buf = encode_point_cloud_binary_fast(
        np.zeros((n, 3), np.float32), np.zeros((n, 3), np.uint8), 0, 0, 0.0)
    assert len(buf) == HEADER + n * 15


def test_empty_cloud_is_header_only():
    buf = encode_point_cloud_binary_fast(
        np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), 5, 7, 0.0)
    assert len(buf) == HEADER
    assert struct.unpack_from('<iiif', buf, 0)[:3] == (5, 7, 0)


def test_accepts_non_contiguous_and_wrong_dtype_input():
    pts64 = np.arange(18, dtype=np.float64).reshape(6, 3)[::2]
    cols16 = np.arange(18, dtype=np.int16).reshape(6, 3)[::2]
    assert not pts64.flags['C_CONTIGUOUS']

    *_, got_pts, got_cols = decode(
        encode_point_cloud_binary_fast(pts64, cols16, 1, 1, 0.0))
    assert np.allclose(got_pts, pts64)
    assert np.array_equal(got_cols, cols16.astype(np.uint8))


def test_positions_are_four_byte_aligned_for_the_typed_array_view():
    """Float32Array over the ArrayBuffer needs a multiple-of-4 byte offset."""
    for n in range(0, 5):
        assert HEADER % 4 == 0
        assert (HEADER + n * 12) % 4 == 0
