#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tmux_spawn_worker.sh \
    --workspace <path> \
    --task-id <id> \
    --prompt-file <file> \
    --log-file <file> \
    --status-file <file> \
    --map-file <file> \
    [--session <session-name>] \
    [--window workers] \
    [--split-current] \
    [--runner-script <path>]
EOF
}

session=""
window="workers"
split_current=0
workspace=""
task_id=""
prompt_file=""
log_file=""
status_file=""
map_file=""
runner_script=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      session="${2:-}"
      shift 2
      ;;
    --window)
      window="${2:-}"
      shift 2
      ;;
    --split-current)
      split_current=1
      shift
      ;;
    --workspace)
      workspace="${2:-}"
      shift 2
      ;;
    --task-id)
      task_id="${2:-}"
      shift 2
      ;;
    --prompt-file)
      prompt_file="${2:-}"
      shift 2
      ;;
    --log-file)
      log_file="${2:-}"
      shift 2
      ;;
    --status-file)
      status_file="${2:-}"
      shift 2
      ;;
    --map-file)
      map_file="${2:-}"
      shift 2
      ;;
    --runner-script)
      runner_script="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$workspace" || -z "$task_id" || -z "$prompt_file" || -z "$log_file" || -z "$status_file" || -z "$map_file" ]]; then
  usage >&2
  exit 1
fi

if [[ -z "$runner_script" ]]; then
  runner_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tmux_run_codex_task.sh"
fi

if [[ ! -x "$runner_script" ]]; then
  echo "Runner script not executable: $runner_script" >&2
  exit 1
fi

if [[ ! -f "$prompt_file" ]]; then
  echo "Prompt file not found: $prompt_file" >&2
  exit 1
fi

command -v tmux >/dev/null || { echo "tmux is required but not installed." >&2; exit 1; }
command -v codex >/dev/null || { echo "codex is required but not installed." >&2; exit 1; }

mkdir -p "$(dirname "$log_file")" "$(dirname "$status_file")" "$(dirname "$map_file")"

runner_cmd=(
  "$runner_script"
  --workspace "$workspace"
  --task-id "$task_id"
  --prompt-file "$prompt_file"
  --log-file "$log_file"
  --status-file "$status_file"
)
printf -v shell_cmd '%q ' "${runner_cmd[@]}"

if [[ "$split_current" -eq 1 ]]; then
  if [[ -z "${TMUX:-}" ]]; then
    echo "--split-current requires running inside tmux." >&2
    exit 1
  fi

  current_window_id="$(tmux display-message -p '#{window_id}')"
  pane_id="$(tmux split-window -d -P -F '#{pane_id}' -t "$current_window_id" -c "$workspace" "$shell_cmd")"
  tmux select-layout -t "$current_window_id" tiled >/dev/null 2>&1 || true
  target_ref="$current_window_id"
else
  if [[ -z "$session" ]]; then
    echo "--session is required unless --split-current is used." >&2
    exit 1
  fi

  if ! tmux has-session -t "$session" 2>/dev/null; then
    tmux new-session -d -s "$session" -n orchestrator -c "$workspace"
  fi

  window_exists=0
  if tmux list-windows -t "$session" -F '#{window_name}' | grep -Fxq "$window"; then
    window_exists=1
  fi

  target_window="${session}:${window}"
  if [[ "$window_exists" -eq 1 ]]; then
    pane_id="$(tmux split-window -d -P -F '#{pane_id}' -t "$target_window" -c "$workspace" "$shell_cmd")"
  else
    pane_id="$(tmux new-window -d -P -F '#{pane_id}' -t "$session" -n "$window" -c "$workspace" "$shell_cmd")"
  fi

  tmux select-layout -t "$target_window" tiled >/dev/null 2>&1 || true
  target_ref="$target_window"
fi

printf '%s|%s|%s|%s|%s\n' "$task_id" "$pane_id" "$target_ref" "$log_file" "$prompt_file" >> "$map_file"
printf '%s\n' "$pane_id"
