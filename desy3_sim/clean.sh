#!/bin/bash
set -euo pipefail

# Delete text and PBS log files in this directory only.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
find "$script_dir" -maxdepth 1 -type f \( -name '*.txt' -o -name '*.out' -o -name '*.err' \) -print -delete
