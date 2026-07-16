#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tmux_reap_completed.sh \
    --session <session-name> \
    --status-file <file> \
    --map-file <file> \
    --seen-file <file>
EOF
}

session=""
status_file=""
map_file=""
seen_file=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      session="${2:-}"
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
    --seen-file)
      seen_file="${2:-}"
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

if [[ -z "$session" || -z "$status_file" || -z "$map_file" || -z "$seen_file" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -f "$status_file" || ! -f "$map_file" ]]; then
  exit 0
fi

mkdir -p "$(dirname "$seen_file")"
touch "$seen_file"

while IFS='|' read -r task_id exit_code started_at ended_at log_file; do
  [[ -n "$task_id" ]] || continue
  record="${task_id}|${exit_code}|${started_at}|${ended_at}|${log_file}"
  if grep -Fxq "$record" "$seen_file"; then
    continue
  fi

  pane_id="$(awk -F'|' -v task="$task_id" '$1==task {pane=$2} END {print pane}' "$map_file")"
  action="not-found"

  if [[ -n "$pane_id" ]] && tmux list-panes -a -F '#{pane_id}' 2>/dev/null | grep -Fxq "$pane_id"; then
    pane_window_id="$(tmux display-message -p -t "$pane_id" '#{window_id}' 2>/dev/null || true)"
    tmux kill-pane -t "$pane_id"
    if [[ -n "$pane_window_id" ]]; then
      tmux select-layout -t "$pane_window_id" tiled >/dev/null 2>&1 || true
    fi
    action="killed"
  fi

  printf '%s\n' "$record" >> "$seen_file"
  printf '%s|%s|%s|%s\n' "$task_id" "$exit_code" "$pane_id" "$action"
done < "$status_file"
