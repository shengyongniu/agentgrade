#!/usr/bin/env bash
#
# Deterministic, offline demo for the agentgrade README asset.
#
# Records a short session that shows `agentgrade test` failing the reward
# threshold and naming the responsible agents. Re-render from the committed
# recording with:
#
#   asciinema rec --overwrite -c "bash scripts/demo.sh" assets/demo.cast
#   agg assets/demo.cast assets/demo.gif --font-size 16 --theme monokai --speed 1.4
#
# Run from the repository root so the bundled example entrypoint resolves.

set -u

# Force colored Rich output even though we are not attached to an interactive
# TTY, and keep the import warning out of the recording.
export AGENTGRADE_NO_WARN=1
export FORCE_COLOR=1
export TERM=xterm-256color
export COLUMNS=100

# Use the project virtualenv's CLI without requiring activation.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PATH="$REPO_ROOT/.venv/bin:$PATH"

CONFIG="examples/simple_agent/agentgrade.yaml"

# Type a command out character-by-character for a natural terminal feel.
type_cmd() {
  local prompt="\033[1;32m$\033[0m "
  printf "%b" "$prompt"
  local text="$1"
  for ((i = 0; i < ${#text}; i++)); do
    printf "%s" "${text:$i:1}"
    sleep 0.025
  done
  printf "\n"
  sleep 0.4
}

clear
sleep 0.6

# 1. Show the bundled example config (a Coder -> Critic pipeline).
type_cmd "cat $CONFIG"
sed -n '1,7p' "$CONFIG"
sleep 1.4

# 2. The money shot: the failing test run.
type_cmd "agentgrade test --config $CONFIG"
agentgrade test --config "$CONFIG"
sleep 2.2

# 3. Turn the failure into an actionable, deterministic prompt patch.
type_cmd "agentgrade improve --suggest --config $CONFIG"
agentgrade improve --suggest --config "$CONFIG"
sleep 2.4
