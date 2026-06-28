"""Tests for Juice Shop scoreboard integration."""

from gym_pentest.scoreboard import Challenge, JuiceShopScoreboard, MockScoreboard, VULN_TO_CHALLENGE


def test_parse_challenges():
    sb = JuiceShopScoreboard(None, "http://localhost:3000")  # type: ignore[arg-type]
    text = '{"data": [{"key": "xss1", "name": "XSS", "category": "XSS", "solved": true, "difficulty": 2}]}'
    challenges = sb.parse_challenges(text)
    assert len(challenges) == 1
    assert challenges[0].solved is True


def test_mock_scoreboard_sync():
    sb = MockScoreboard()
    state = sb.sync()
    assert state.solved_count == 0
    sb.mark_solved("xssChallenge")
    state = sb.sync()
    assert state.solved_count == 1
    assert "xssChallenge" in state.newly_solved


def test_scoreboard_reward_in_env(mock_env):
    mock_env.reset(seed=0)
    _, reward, _, _, info = mock_env.step(4)  # XSS
    assert info["challenges_solved"] >= 1
    assert reward > 30.0


def test_vuln_to_challenge_mapping():
    assert VULN_TO_CHALLENGE["XSS"] == "xssChallenge"
    assert VULN_TO_CHALLENGE["SQLi"] == "sqliChallenge"
