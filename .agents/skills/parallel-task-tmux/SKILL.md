---
name: parallel-task-tmux
description: >
  Only to be triggered by explicit /parallel-task-tmux commands. Dependency-aware parallel plan execution that launches each unblocked Codex worker in a live tmux pane, keeps pane output visible in real time, and shrinks the layout by closing panes as tasks complete.
---

# Parallel Task Executor (tmux Live Workers)

Run dependency-ordered plan execution like `parallel-task`, but route each worker through `codex exec` inside tmux panes so progress is visible in real time.

Use this skill when you want:
- Live visibility into each running worker
- A tiled worker view that expands as tasks launch
- Automatic pane cleanup that shrinks the layout as tasks finish

## Execution Model

1. Parse the plan and compute unblocked tasks by `depends_on`.
2. Build one worker prompt file per task.
3. Launch one tmux pane per task via `scripts/tmux_spawn_worker.sh`.
4. Monitor status/logs and reap completed panes via `scripts/tmux_reap_completed.sh`.
5. Validate completed tasks before advancing to the next dependency wave.
6. Repeat until all requested tasks are complete.

## Preflight

Run these checks before launching workers:

```bash
command -v tmux >/dev/null
command -v codex >/dev/null
```

Resolve the skill directory once so helper script paths are stable:

```bash
SKILL_ROOT="$(fd -td -p 'parallel-task-tmux' "$PWD" "$HOME/.codex/skills" "$HOME/.agents/skills" 2>/dev/null | head -n 1)"
```

Fail fast if `SKILL_ROOT` is empty.

Initialize one run directory per execution:

```bash
PLAN_FILE="./plan.md"
PLAN_BASE="$(basename "$PLAN_FILE" .md)"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
SESSION_NAME="swarms-${PLAN_BASE}-${RUN_ID}"
RUN_ROOT=".swarms/tmux/${SESSION_NAME}"
mkdir -p "$RUN_ROOT/prompts" "$RUN_ROOT/logs"
: > "$RUN_ROOT/status.tsv"
: > "$RUN_ROOT/task_map.tsv"
: > "$RUN_ROOT/reaped.tsv"
```

## Plan Parsing and Scheduling

Mirror `parallel-task` behavior:
- Parse task sections (for example `### T1:`).
- Extract task id, title, `depends_on`, location, description, acceptance criteria, validation.
- Filter to requested subset and required dependencies if the user passes a subset.
- Launch only tasks whose dependencies are complete.

## Worker Prompt Contract

Create one prompt file per task at:
- `$RUN_ROOT/prompts/<TASK_ID>.md`

Use the same task context and completion requirements as `parallel-task`:
- Read the working plan and relevant files before coding.
- Default to TDD RED phase first using a `tdd_test_writer` subagent, or explicitly record `reason_not_testable` with an alternative verification contract.
- Treat RED-phase tests (or approved non-testable verification plan) as the implementation contract.
- Implement acceptance criteria.
- Keep edits atomic.
- Run the exact new/updated test commands until GREEN, or run the documented alternative verification and capture evidence.
- Commit only task-scoped files.
- Update the plan status/log/files after the commit.
- Return modified files, criteria coverage, and RED -> GREEN or non-testable verification evidence.

## Launching tmux Workers

When running inside tmux, prefer in-place splits so the user can keep chatting in the current Codex pane while workers appear beside it.

In-place mode (recommended inside tmux):

```bash
"$SKILL_ROOT/scripts/tmux_spawn_worker.sh" \
  --split-current \
  --workspace "$PWD" \
  --task-id "$TASK_ID" \
  --prompt-file "$RUN_ROOT/prompts/${TASK_ID}.md" \
  --log-file "$RUN_ROOT/logs/${TASK_ID}.log" \
  --status-file "$RUN_ROOT/status.tsv" \
  --map-file "$RUN_ROOT/task_map.tsv"
```

Detached session mode (for non-tmux shells):

```bash
"$SKILL_ROOT/scripts/tmux_spawn_worker.sh" \
  --session "$SESSION_NAME" \
  --workspace "$PWD" \
  --task-id "$TASK_ID" \
  --prompt-file "$RUN_ROOT/prompts/${TASK_ID}.md" \
  --log-file "$RUN_ROOT/logs/${TASK_ID}.log" \
  --status-file "$RUN_ROOT/status.tsv" \
  --map-file "$RUN_ROOT/task_map.tsv"
```

Behavior:
- In split-current mode, launches workers as new panes in the current tmux window.
- In detached mode, creates tmux session/window if missing.
- Re-tiles panes after each launch so the grid expands live.
- Streams `codex exec` output live in each pane while also writing logs to `logs/<TASK_ID>.log`.
- Tracks pane ids in `task_map.tsv` for deterministic cleanup.

In detached mode, if not already attached, print:

```bash
tmux attach -t "$SESSION_NAME"
```

## Monitoring and Pane Reaping

During each wave, monitor active panes:

```bash
tmux list-panes -t "${SESSION_NAME}:workers" -F '#{pane_id} #{pane_current_command} #{pane_dead}'
```

Collect completed workers and close their panes:

```bash
"$SKILL_ROOT/scripts/tmux_reap_completed.sh" \
  --session "$SESSION_NAME" \
  --status-file "$RUN_ROOT/status.tsv" \
  --map-file "$RUN_ROOT/task_map.tsv" \
  --seen-file "$RUN_ROOT/reaped.tsv"
```

This is what gives the expand/shrink behavior:
- Launching tasks adds panes and expands the tile grid.
- Reaping completed tasks kills panes and shrinks the grid.

Inspect logs before accepting a task:

```bash
tail -n 80 "$RUN_ROOT/logs/${TASK_ID}.log"
```

## Validation and Wave Control

For each completed task:
1. Verify files changed as expected.
2. Verify plan status/log/files fields were updated.
3. Verify RED -> GREEN test evidence or explicit `reason_not_testable` plus concrete alternative verification output.
4. Mark task complete in orchestrator state only after verification passes.
5. Retry failed tasks with corrected prompts when needed.

Launch the next wave only when all tasks in the current wave are verified complete.

## Finalization

After all tasks are complete:
1. Run final validation commands from the plan.
2. Confirm no pending tasks remain.
3. Provide an execution summary with completed, failed, retried tasks, modified files, and verification evidence.

Leave the tmux session alive by default for auditability. Only kill it if the user asks:

```bash
tmux kill-session -t "$SESSION_NAME"
```

## Error Handling

- Missing task ids in subset: list valid task ids from the parsed plan.
- `tmux` unavailable: stop and instruct user to install tmux.
- Worker exits non-zero: inspect log, repair prompt/context, relaunch task.
- No status updates: verify pane is still running; if dead without status line, relaunch.
- Parse failure: report parser assumptions and request corrected plan format.

## Example Usage

```text
/parallel-task-tmux plan.md
/parallel-task-tmux ./plans/auth-plan.md T1 T2 T4
/parallel-task-tmux user-profile-plan.md --tasks T3 T7
```
