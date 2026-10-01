"""Subprocess deadlines for the test scripts.

MIL_HWX_TEST_TIMEOUT_SCALE multiplies every deadline, for slow hosts such as
qemu-user. An unset, unparsable, non-finite, zero or negative value means 1.0.
A scaled deadline is capped at MAX_SECONDS: subprocess waits through
select.poll(), which raises OverflowError above 2**31 - 1 milliseconds.
"""
import math
import os

MAX_SECONDS = 2_147_483


def _scale():
    try:
        scale = float(os.environ.get("MIL_HWX_TEST_TIMEOUT_SCALE", "1"))
    except ValueError:
        return 1.0
    return scale if math.isfinite(scale) and scale > 0 else 1.0


SCALE = _scale()


def scaled_timeout(seconds):
    return min(seconds * SCALE, MAX_SECONDS)
