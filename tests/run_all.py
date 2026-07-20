"""Run the whole test suite: `.venv/bin/python -m tests.run_all`"""
import sys

from tests import runner


def main() -> int:
    from tests import test_gradients, test_invariants, test_training
    return runner.main([test_gradients, test_invariants, test_training])


if __name__ == "__main__":
    sys.exit(main())
