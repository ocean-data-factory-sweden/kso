#!/bin/bash
set -e

exec jupyter lab --ip=0.0.0.0 --no-browser "$@"
