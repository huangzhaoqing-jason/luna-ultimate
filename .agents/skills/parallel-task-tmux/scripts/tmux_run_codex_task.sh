#!/usr/bin/env bash
set -u -o pipefail

usage() {
  cat <<'EOF'
Usage:
  tmux_run_codex_task.sh \
    --workspace <path> \
    --task-id <id> \
    --prompt-file <file> \
    --log-file <file> \
    --status-file <file>
EOF
}

workspace=""
task_id=""
prompt_file=""
log_file=""
status_file=""

while [[ $# -gt 0 ]]; do
  case "$1" in
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

if [[ -z "$workspace" || -z "$task_id" || -z "$prompt_file" || -z "$log_file" || -z "$status_file" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -f "$prompt_file" ]]; then
  echo "Prompt file not found: $prompt_file" >&2
  exit 1
fi

mkdir -p "$(dirname "$log_file")" "$(dirname "$status_file")"

start_ts="$(date -Is)"
rc=0

(
  cd "$workspace" || exit 1
  codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check --cd "$workspace" - < "$prompt_file" 2>&1 | tee "$log_file"
) || rc=$?

end_ts="$(date -Is)"
printf '%s|%s|%s|%s|%s\n' "$task_id" "$rc" "$start_ts" "$end_ts" "$log_file" >> "$status_file"

exit "$rc"
