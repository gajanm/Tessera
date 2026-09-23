"""Shared fixtures. Everything here runs on CPU; nothing loads Pi3X or SAM3."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakePipeline:
    """Stand-in for IncrementalPi3 that records what the server asked of it."""

    def __init__(self):
        self.total_points = 1234
        self.chunks_processed = 7
        self.chunk_size = 16
        self.overlap = 6
        self.conf_thre = 0.05
        self.num_workers = 1
        self.frame_buffer = []
        self.frame_data = []
        self.global_points = []
        self.configured = None
        self.reset_calls = 0
        self.save_calls = 0

    def configure(self, **kwargs):
        self.configured = kwargs

    def reset(self):
        self.reset_calls += 1

    def save_to_disk(self):
        self.save_calls += 1

    def frames_until_ready(self):
        return 3

    def get_global_cloud(self, max_points=100_000):
        import numpy as np
        return {'points': np.zeros((0, 3), np.float32),
                'colors': np.zeros((0, 3), np.uint8)}


@pytest.fixture
def pipe():
    """A real IncrementalPi3 on CPU with no model loaded."""
    from pipeline import IncrementalPi3
    return IncrementalPi3(device='cpu')


@pytest.fixture
def client(monkeypatch):
    """TestClient with a stubbed pipeline and no background worker thread."""
    from fastapi.testclient import TestClient
    import server

    fake = FakePipeline()
    monkeypatch.setattr(server, 'pipeline', fake)
    monkeypatch.setattr(server, 'processing_loop', lambda loop: None)
    with TestClient(server.app) as c:
        c.fake = fake
        yield c
