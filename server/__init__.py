"""
Patch: add __iter__ to StepResult so it also supports tuple unpacking.
This is monkey-patched into env.py's StepResult class at import time.
"""

# This is handled directly inside env.py — this file is kept for reference only.
