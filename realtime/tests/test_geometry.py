"""Sim3 alignment and image-sizing math. Pure tensor work, no GPU."""
import pytest
import torch

from pipeline import PATCH_SIZE, PIXEL_LIMIT, compute_target_size


def _rotation(seed):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    q = q @ torch.diag(torch.sign(torch.diagonal(r)))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q.float()


def _decompose(transform):
    sr = transform[0, :3, :3]
    scale = torch.det(sr).abs().pow(1 / 3)
    return scale.item(), sr / scale, transform[0, :3, 3]


# ── compute_target_size ──────────────────────────────────────────

@pytest.mark.parametrize("w,h", [(1920, 1080), (1280, 720), (640, 480),
                                 (480, 640), (3840, 2160), (100, 100)])
def test_target_size_is_patch_aligned_and_bounded(w, h):
    tw, th = compute_target_size(w, h)
    assert tw % PATCH_SIZE == 0 and th % PATCH_SIZE == 0
    assert 0 < tw * th <= PIXEL_LIMIT


@pytest.mark.parametrize("w,h", [(1920, 1080), (1280, 720), (640, 480)])
def test_target_size_roughly_preserves_aspect(w, h):
    tw, th = compute_target_size(w, h)
    assert abs((tw / th) - (w / h)) / (w / h) < 0.12


# ── Sim3 ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("scale", [0.1, 1.0, 2.5, 50.0])
def test_recovers_a_known_sim3(pipe, scale):
    g = torch.Generator().manual_seed(7)
    rot = _rotation(1)
    trans = torch.randn(3, generator=g) * 5
    src = torch.randn(1, 4000, 3, generator=g) * 3
    tgt = scale * (src @ rot.T) + trans
    ones = torch.ones(1, 4000, dtype=torch.bool)

    got_s, got_r, got_t = _decompose(
        pipe._compute_sim3_umeyama_masked(src, tgt, ones, ones))

    assert got_s == pytest.approx(scale, rel=1e-4)
    assert torch.allclose(got_r, rot, atol=1e-4)
    assert torch.allclose(got_t, trans, atol=1e-3)


def test_masked_out_points_do_not_influence_the_fit(pipe):
    """Garbage under the mask must not move the answer."""
    g = torch.Generator().manual_seed(11)
    rot, scale = _rotation(2), 1.7
    trans = torch.tensor([1.0, -2.0, 0.5])
    src = torch.randn(1, 3000, 3, generator=g)
    tgt = scale * (src @ rot.T) + trans

    mask = torch.ones(1, 3000, dtype=torch.bool)
    mask[0, ::3] = False
    tgt = tgt.clone()
    tgt[0, ::3] = 1e4                      # nonsense, but masked off

    got_s, got_r, got_t = _decompose(
        pipe._compute_sim3_umeyama_masked(src, tgt, mask, mask))

    assert got_s == pytest.approx(scale, rel=1e-4)
    assert torch.allclose(got_r, rot, atol=1e-4)
    assert torch.allclose(got_t, trans, atol=1e-3)


def test_too_few_correspondences_falls_back_to_identity(pipe):
    src = torch.randn(1, 5, 3)
    tgt = torch.randn(1, 5, 3)
    ones = torch.ones(1, 5, dtype=torch.bool)
    out = pipe._compute_sim3_umeyama_masked(src, tgt, ones, ones)
    assert torch.allclose(out[0], torch.eye(4))


def test_mirrored_target_does_not_yield_a_reflection(pipe):
    """A reflection has det -1 and would flip the reconstruction."""
    g = torch.Generator().manual_seed(3)
    src = torch.randn(1, 3000, 3, generator=g)
    tgt = src.clone()
    tgt[..., 2] *= -1
    ones = torch.ones(1, 3000, dtype=torch.bool)

    _, rot, _ = _decompose(pipe._compute_sim3_umeyama_masked(src, tgt, ones, ones))
    assert torch.det(rot).item() == pytest.approx(1.0, abs=1e-4)


def test_identical_clouds_give_identity(pipe):
    g = torch.Generator().manual_seed(5)
    pts = torch.randn(1, 2000, 3, generator=g)
    ones = torch.ones(1, 2000, dtype=torch.bool)
    out = pipe._compute_sim3_umeyama_masked(pts, pts, ones, ones)
    assert torch.allclose(out[0], torch.eye(4), atol=1e-4)


def test_apply_sim3_maps_source_onto_target(pipe):
    g = torch.Generator().manual_seed(13)
    rot, scale = _rotation(4), 0.8
    trans = torch.tensor([3.0, 1.0, -1.0])
    src = torch.randn(1, 2, 4, 5, 3, generator=g)
    tgt = scale * (src.reshape(1, -1, 3) @ rot.T) + trans

    flat = src.reshape(1, -1, 3)
    ones = torch.ones(1, flat.shape[1], dtype=torch.bool)
    transform = pipe._compute_sim3_umeyama_masked(flat, tgt, ones, ones)

    out = pipe._apply_sim3_to_points(src, transform)
    assert out.shape == src.shape
    assert torch.allclose(out.reshape(1, -1, 3), tgt, atol=1e-3)


def test_apply_sim3_to_poses_translates_correctly(pipe):
    scale = 2.0
    rot = _rotation(6)
    trans = torch.tensor([1.0, 2.0, 3.0])
    transform = torch.eye(4).unsqueeze(0)
    transform[0, :3, :3] = scale * rot
    transform[0, :3, 3] = trans

    poses = torch.eye(4).repeat(1, 3, 1, 1)
    poses[0, :, :3, 3] = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])

    out = pipe._apply_sim3_to_poses(poses, transform)
    expected = scale * (poses[0, :, :3, 3] @ rot.T) + trans
    assert torch.allclose(out[0, :, :3, 3], expected, atol=1e-4)


@pytest.mark.xfail(reason="known: sim3 @ pose leaves scale in the rotation block, "
                          "so the result is not orthonormal", strict=True)
def test_apply_sim3_to_poses_keeps_rotation_orthonormal(pipe):
    transform = torch.eye(4).unsqueeze(0)
    transform[0, :3, :3] = 2.0 * _rotation(6)
    poses = torch.eye(4).repeat(1, 2, 1, 1)

    out = pipe._apply_sim3_to_poses(poses, transform)
    rot = out[0, 0, :3, :3]
    assert torch.det(rot).item() == pytest.approx(1.0, abs=1e-4)
