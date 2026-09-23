"""HTTP endpoints and the viewer broadcast path, against a stubbed pipeline."""
import asyncio
import queue

import pytest

import server


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Keep tests off the real filesystem, the real key and any leftover state."""
    monkeypatch.setattr(server, '_wipe_saved_state', lambda: None)
    monkeypatch.setattr(server, '_openai_client', None)
    monkeypatch.setenv('OPENAI_API_KEY', '')
    server.connected_viewers.clear()
    yield
    server.connected_viewers.clear()


class RecordingWS:
    def __init__(self, name, fail=False, on_send=None):
        self.name, self.fail, self.on_send = name, fail, on_send
        self.sent = []

    async def send_json(self, message):
        await asyncio.sleep(0)
        if self.on_send:
            self.on_send()
        if self.fail:
            raise RuntimeError("socket closed")
        self.sent.append(message)

    async def send_bytes(self, data):
        await asyncio.sleep(0)
        if self.fail:
            raise RuntimeError("socket closed")
        self.sent.append(data)

    def __hash__(self):
        return hash(self.name)


# ── endpoints ────────────────────────────────────────────────────

def test_health_reports_pipeline_state(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["total_points"] == 1234
    assert body["chunks_processed"] == 7


def test_stats_exposes_queue_capacity(client):
    body = client.get("/stats").json()
    assert body["queue_max"] == server.frame_queue.maxsize
    assert body["total_points"] == 1234


def test_configure_passes_values_through_and_echoes_them(client):
    body = client.post("/configure", json={"chunk_size": 12, "overlap": 4}).json()
    assert client.fake.configured == {"chunk_size": 12, "overlap": 4, "conf_thre": None}
    assert body["status"] == "configured"


def test_get_config_reports_current_values(client):
    body = client.get("/config").json()
    assert body["chunk_size"] == 16 and body["overlap"] == 6


def test_reset_clears_the_queue_and_resets_the_pipeline(client):
    server.frame_queue.put_nowait(object())
    assert server.frame_queue.qsize() == 1

    assert client.post("/reset").json()["status"] == "reset_complete"
    assert client.fake.reset_calls == 1
    assert server.frame_queue.empty()


def test_save_refuses_when_there_is_nothing_to_save(client):
    assert client.post("/save").status_code == 400
    assert client.fake.save_calls == 0


def test_save_persists_when_there_is_data(client):
    client.fake.global_points = [object()]
    assert client.post("/save").json()["status"] == "saved"
    assert client.fake.save_calls == 1


def test_analyze_without_labeled_objects_explains_itself(client):
    body = client.post("/analyze", json={"question": "where is the desk?"}).json()
    assert "label" in body["answer"].lower()


def test_llm_endpoints_fail_cleanly_without_a_key(client):
    payload = {"question": "q", "objects": [{"label": "desk"}]}
    assert client.post("/analyze", json=payload).status_code == 500
    assert client.post("/parse-nl-prompt", json={"text": "a chair"}).status_code == 500


def test_label_objects_needs_frames(client):
    assert client.post("/label-objects", json={"prompts": ["chair"]}).status_code == 400


def test_label_objects_rejects_a_second_concurrent_run(client, monkeypatch):
    client.fake.frame_data = [{"frame_idx": 0}]
    monkeypatch.setattr(server, '_labeling_in_progress', True)
    r = client.post("/label-objects", json={"prompts": ["chair"]})
    assert r.status_code == 409


# ── broadcast ────────────────────────────────────────────────────

def test_broadcast_survives_a_viewer_joining_mid_send():
    """A viewer connecting during an await used to raise 'set changed size'."""
    def join_late():
        server.connected_viewers.add(RecordingWS("late"))

    server.connected_viewers.update(
        {RecordingWS("a", on_send=join_late), RecordingWS("b")})
    asyncio.run(server.broadcast_to_viewers({"type": "ping"}))


def test_broadcast_survives_a_viewer_leaving_mid_send():
    victim = RecordingWS("victim")

    def leave():
        server.connected_viewers.discard(victim)

    server.connected_viewers.update({RecordingWS("a", on_send=leave), victim})
    asyncio.run(server.broadcast_to_viewers({"type": "ping"}))


def test_failed_sockets_are_dropped_and_healthy_ones_still_receive():
    good, bad = RecordingWS("good"), RecordingWS("bad", fail=True)
    server.connected_viewers.update({good, bad})

    asyncio.run(server.broadcast_to_viewers({"type": "ping"}))

    assert good.sent == [{"type": "ping"}]
    assert bad not in server.connected_viewers
    assert good in server.connected_viewers


def test_binary_broadcast_reaches_every_viewer():
    a, b = RecordingWS("a"), RecordingWS("b")
    server.connected_viewers.update({a, b})
    asyncio.run(server.broadcast_binary(b"\x00\x01"))
    assert a.sent == [b"\x00\x01"] and b.sent == [b"\x00\x01"]


def test_broadcast_to_nobody_is_a_noop():
    asyncio.run(server.broadcast_to_viewers({"type": "ping"}))


def test_viewers_are_sent_concurrently_not_one_after_another():
    """A slow viewer must not hold up the ones behind it."""
    order = []

    class Tracked(RecordingWS):
        async def send_json(self, message):
            order.append(("start", self.name))
            await asyncio.sleep(0.02)
            order.append(("end", self.name))

    server.connected_viewers.update({Tracked("a"), Tracked("b"), Tracked("c")})
    asyncio.run(server.broadcast_to_viewers({"type": "ping"}))

    assert [k for k, _ in order[:3]] == ["start", "start", "start"]
