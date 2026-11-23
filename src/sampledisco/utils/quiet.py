"""Scoped warning suppression.

A module-level ``warnings.filterwarnings("ignore")`` runs at import and hides every
later warning in the session. Decorate the noisy entry points with
``quiet_warnings`` instead: warnings are ignored only while that call runs.
"""
import functools
import warnings


def quiet_warnings(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*args, **kwargs)
    return wrapped
