"""Tests for vulnerability evidence tracking."""

from gym_pentest.evidence import EvidenceTracker


def test_add_finding():
    tracker = EvidenceTracker()
    evidence = tracker.add("XSS", "http://localhost/rest/feedback", "<script>", 201)
    assert evidence is not None
    assert tracker.count == 1


def test_duplicate_finding_rejected():
    tracker = EvidenceTracker()
    tracker.add("XSS", "http://localhost/rest/feedback", "<script>", 201)
    dup = tracker.add("XSS", "http://localhost/rest/feedback", "<script>", 201)
    assert dup is None
    assert tracker.count == 1


def test_evidence_in_env_info(mock_env):
    mock_env.reset(seed=0)
    mock_env.step(4)
    _, _, _, _, info = mock_env.step(0)
    assert info["vulnerabilities"] >= 1
    assert len(info["evidence"]) >= 1
