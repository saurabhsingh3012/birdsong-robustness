"""Tests for the validation harness itself.

The harness is what turns "the code looks right" into "requested equals measured", so it has to
be trustworthy in its own right. Two things are checked: that the pass/fail logic is honest --
including the one-sided checks, where a naive absolute-error rule would pass a violated
requirement -- and that a full run over the real degradation pipeline comes back clean, which is
the evidence quoted in the README.
"""

from __future__ import annotations

import json

from birdsong_robustness import validate
from birdsong_robustness.validate import Check, ValidationReport, run_validation


def test_two_sided_check_pass_and_fail():
    within = Check("a", "c", "q", "dB", requested=0.0, measured=0.04, tolerance=0.05)
    outside = Check("a", "c", "q", "dB", requested=0.0, measured=0.06, tolerance=0.05)
    assert within.passed
    assert not outside.passed


def test_at_least_check_is_one_sided():
    """A '>= 80 dB' requirement must pass at 102 dB and fail at 5 dB.

    A two-sided ``|measured - 80| <= 80`` rule would pass both, which is exactly the bug the
    direction field exists to prevent for the resample stopband check.
    """
    ample = Check(
        "a", "c", "q", "dB", requested=80.0, measured=102.0, tolerance=0.5, direction="at-least"
    )
    shortfall = Check(
        "a", "c", "q", "dB", requested=80.0, measured=5.0, tolerance=0.5, direction="at-least"
    )
    assert ample.passed
    assert not shortfall.passed


def test_at_most_check_is_one_sided():
    ok = Check(
        "a", "c", "q", "u", requested=0.05, measured=0.01, tolerance=0.0, direction="at-most"
    )
    bad = Check(
        "a", "c", "q", "u", requested=0.05, measured=0.2, tolerance=0.0, direction="at-most"
    )
    assert ok.passed
    assert not bad.passed


def test_comparator_symbols():
    assert Check("a", "c", "q", "u", 0, 0, 0).comparator == "=="
    assert Check("a", "c", "q", "u", 0, 0, 0, direction="at-least").comparator == ">="
    assert Check("a", "c", "q", "u", 0, 0, 0, direction="at-most").comparator == "<="


def test_report_worst_ignores_one_sided_checks():
    """'Worst error' is meaningless for a one-sided check, so it must be excluded from that stat."""
    report = ValidationReport(
        checks=(
            Check("a", "c", "q", "dB", 0.0, 0.02, 0.05),
            Check("a", "c", "q", "dB", 80.0, 500.0, 0.5, direction="at-least"),
        ),
        observations=(),
    )
    worst = report.worst
    assert worst is not None
    assert worst.direction == "two-sided"


def test_report_serialises_with_directions():
    report = run_validation(suites=["noise models"])
    payload = json.loads(report.to_json())
    assert payload["n_checks"] > 0
    assert all("direction" in c for c in payload["checks"])


def test_noise_suite_passes():
    """The coloured-noise slopes are analytic targets; this suite must come back clean."""
    report = run_validation(suites=["noise models"])
    assert report.n_failed == 0
    assert len(report.checks) > 0


def test_full_validation_passes():
    """The headline result: every degradation in the pipeline measures back what was requested.

    This is the number the README quotes. If it ever regresses, that claim is false, so the test
    asserts zero failures across the entire suite rather than a threshold.
    """
    report = run_validation()
    assert report.n_failed == 0, (
        f"{report.n_failed} checks outside tolerance; worst: {report.worst}"
    )
    # A broad run, not a token one -- guards against a suite silently going missing.
    assert len(report.checks) >= 100


def test_main_returns_zero_on_success(capsys):
    """The console entry point must exit 0 when everything passes, for CI to key on."""
    code = validate.main(["--no-observations"])
    captured = capsys.readouterr()
    assert code == 0
    assert "NO CLASSIFIER HAS BEEN EVALUATED" in captured.out
    assert "within tolerance" in captured.out
