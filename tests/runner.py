"""Minimal test runner — collects `test_*` functions and reports pass/fail.

The repo has no third-party dependencies and the tests keep it that way. Test
functions are named `test_*` and take no arguments, so `pytest tests/` also
works unchanged for anyone who has it installed.
"""
import traceback
import types


def run_module(module: types.ModuleType, verbose: bool = True) -> tuple[int, int]:
    """Run every `test_*` function in `module`. Returns (passed, failed)."""
    tests = [
        (name, fn)
        for name, fn in sorted(vars(module).items())
        if name.startswith("test_") and callable(fn)
    ]
    passed = failed = 0
    print(f"\n{module.__name__}  ({len(tests)} tests)")
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            if verbose:
                print("        " + traceback.format_exc().replace("\n", "\n        "))
        else:
            passed += 1
            print(f"  ok    {name}")
    return passed, failed


def main(modules: list[types.ModuleType]) -> int:
    """Run several modules; returns a process exit code."""
    total_pass = total_fail = 0
    for module in modules:
        p, f = run_module(module)
        total_pass += p
        total_fail += f

    print(f"\n{'=' * 60}")
    print(f"{total_pass} passed, {total_fail} failed")
    return 1 if total_fail else 0
