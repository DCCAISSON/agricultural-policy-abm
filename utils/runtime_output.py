from __future__ import annotations

import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Iterator


@contextmanager
def suppress_runtime_output(enabled: bool = True) -> Iterator[None]:
    """
    Silence noisy model/runtime prints while keeping script-level progress messages.

    Args:
        enabled: When False, stdout/stderr pass through unchanged.
    """
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield
