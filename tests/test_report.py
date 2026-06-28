"""Tests for finding report generation."""

from gym_pentest.report import (
    compute_risk_score,
    generate_finding_reports,
    reports_to_markdown,
)


def test_compute_risk_score_confirmed_higher():
    base = compute_risk_score("XSS", confirmed=False, status_code=200)
    confirmed = compute_risk_score("XSS", confirmed=True, status_code=200)
    assert confirmed > base


def test_generate_finding_reports():
    evidence = [
        {
            "evidence_id": "abc123",
            "vuln_type": "XSS",
            "endpoint": "http://localhost:3000/rest/feedback",
            "payload": "<script>alert(1)</script>",
            "status_code": 200,
            "confirmed": True,
        }
    ]
    reports = generate_finding_reports(evidence)
    assert len(reports) == 1
    assert reports[0].owasp_category.startswith("A03")
    assert reports[0].severity == "High"


def test_reports_to_markdown():
    evidence = [
        {
            "evidence_id": "abc123",
            "vuln_type": "SQLi",
            "endpoint": "http://localhost:3000/rest/user/login",
            "payload": "' OR 1=1--",
            "status_code": 500,
            "confirmed": False,
        }
    ]
    reports = generate_finding_reports(evidence)
    md = reports_to_markdown(reports)
    assert "Vulnerability Assessment Report" in md
    assert "SQLi" in md
    assert "Remediation" in md
