"""Tests for environment-sensitive snapshot runtime handling."""

from test_common import is_lsan_ptrace_error


def test_detects_lsan_ptrace_error():
    """LeakSanitizer ptrace failures should be detected explicitly."""
    stderr = (
        "==36==LeakSanitizer has encountered a fatal error.\n"
        "==36==HINT: LeakSanitizer does not work under ptrace (strace, gdb, etc)\n"
    )

    assert is_lsan_ptrace_error(stderr) is True


def test_ignores_other_sanitizer_output():
    """Only the ptrace-specific LeakSanitizer failure should trigger a skip."""
    stderr = "ERROR: AddressSanitizer: heap-use-after-free"

    assert is_lsan_ptrace_error(stderr) is False
