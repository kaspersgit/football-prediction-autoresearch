#!/usr/bin/env bash
# Run predictions for the current game week across all tracked leagues.
# Default threshold is 0.04 (production guard — only bets with >=4% edge).
# Usage:
#   ./predict.sh                      # threshold 0.04
#   ./predict.sh --threshold 0.0      # all value bets (research/exploration)
#   ./predict.sh --threshold 0.05     # stricter filter
set -euo pipefail
cd "$(dirname "$0")"

# Inject --threshold 0.04 unless caller already passed --threshold
if [[ ! " $* " =~ " --threshold " ]]; then
  set -- "$@" --threshold 0.04
fi

uv run python main.py --predict "$@" 2>/dev/null
