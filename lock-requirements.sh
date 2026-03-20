#!/bin/sh

set -e

rm -f requirements.txt
rm -f requirements-dev.txt

# Save normal end user requirements.
# Also save the build deps (i.e. setuptools).
pip-compile \
    --all-build-deps \
    --output-file requirements.txt \
    setup.py

# Save dev requirements
# (i.e. including nose2, coverage, ...).
# Also includes the normal requirements.
# Also save the build deps (i.e. setuptools).
pip-compile \
    --all-build-deps \
    --output-file requirements-dev.txt \
    --extra dev \
    setup.py

