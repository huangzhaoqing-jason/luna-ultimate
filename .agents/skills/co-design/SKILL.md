---
name: co-design
description: >
  Only to be triggered by explicit /co-design commands. Parallel task executor that routes frontend/design tasks to Claude CLI print mode for superior design output.
---

# Co-Design: Parallel Task Executor with Design-Mode Routing

You are an Orchestrator for subagents. Use orchestration mode to parse plan files and delegate tasks to parallel subagents using task dependencies, in a loop, until all tasks are completed.

**Key difference from standard parallel-task**: When a task involves **frontend design, styling, UI, or visual work**, you launch it via `claude -p` (print mode) in a background shell instead of using the Task tool. This gives the design agent full CLI capabilities and tool access. All other tasks use normal Task tool subagents.

## Task Classification

Before launching each task, classify it as **design** or **standard**:

### Design Tasks (route to `claude -p`)
A task is a **design task** if it primarily involves ANY of:
- CSS, SCSS, Tailwind, or any styling
- HTML structure/templates for UI
- React/Vue/Svelte/Angular component **markup and styling**
- Layout, responsive design, grid, flexbox
- Design tokens, theme files, color schemes
- UI component creation (buttons, cards, modals, navbars, etc.)
- Animation, transitions, visual effects
- Accessibility (a11y) related to visual presentation
- Asset management (icons, images, fonts)
- Design system implementation

### Standard Tasks (route to Task tool subagent)
Everything else:
- Backend logic, API endpoints, database
- Business logic, services, utilities
- Configuration, tooling, CI/CD
- Testing (unless it's visual/snapshot testing)
- State management, data fetching
- Authentication, authorization

**When in doubt**: If a task mixes both (e.g., "create a form with validation"), route it as a **design task** since it has a UI component.

## Process

### Step 1: Parse Request

Extract from user request:
1. **Plan file**: The markdown plan to read
2. **Task subset** (optional): Specific task IDs to run

If no subset provided, run the full plan.

### Step 2: Read & Parse Plan

1. Find task subsections (e.g., `### T1:` or `### Task 1.1:`)
2. For each task, extract:
   - Task ID and name
   - **depends_on** list (from `- **depends_on**: [...]`)
   - Full content (description, location, acceptance criteria, validation)
3. Build task list
4. **Classify each task** as `design` or `standard` based on the rules above
5. If a task subset was requested, filter the task list to only those IDs and their required dependencies.

### Step 3: Launch Tasks

For each **unblocked** task, launch it based on its classification:

#### For STANDARD tasks → Use Task tool subagent

Launch with:
- **description**: "Implement task [ID]: [name]"
- **prompt**: Use the Task Prompt Template below
- **subagent_type**: "parallel"

#### For DESIGN tasks → Use `claude -p` in background shell

Construct the exact same prompt you would give a subagent, then launch via Bash:

```bash
claude -p "YOUR_PROMPT_HERE" \
  --allowedTools "Bash,Read,Edit,Write,Glob,Grep,WebFetch,WebSearch" \
  --max-turns 50 \
  --output-format text \
  > /tmp/co-design-[TASK_ID]-output.log 2>&1 &
```

**CRITICAL rules for design task launch:**
1. The prompt content must be IDENTICAL to what you'd give a Task tool subagent (use the same Task Prompt Template)
2. Escape the prompt properly for shell execution - use a heredoc approach:
   ```bash
   claude -p "$(cat <<'PROMPT_EOF'
   [full prompt content here]
   PROMPT_EOF
   )" \
     --allowedTools "Bash,Read,Edit,Write,Glob,Grep,WebFetch,WebSearch" \
     --max-turns 50 \
     --output-format text \
     > /tmp/co-design-[TASK_ID]-output.log 2>&1 &
   ```
3. Log output to `/tmp/co-design-[TASK_ID]-output.log` for later inspection
4. Track the background PID so you can check completion
5. Launch ALL unblocked design tasks in parallel (multiple background shells)

Launch all unblocked tasks (both types) in parallel. A task is unblocked if all IDs in its depends_on list are complete.

### Task Prompt Template

Use this for BOTH standard subagents AND design-mode `claude -p` tasks:

```
You are implementing a specific task from a development plan.

## Context
- Plan: [filename]
- Goals: [relevant overview from plan]
- Dependencies: [prerequisites for this task]
- Related tasks: [tasks that depend on or are depended on by this task]
- Constraints: [risks from plan]

## Your Task
**Task [ID]: [Name]**

Location: [File paths]
Description: [Full description]

Acceptance Criteria:
[List from plan]

Validation:
[Tests or verification from plan]

## Instructions
1. Read the working plan and fully understand this task before coding.
2. Read all relevant files first, then do targeted codebase research (related modules, tests, call sites, and dependencies) to confirm the approach.
3. Default to TDD RED phase first using a `tdd_test_writer` subagent:
   - Pass task context and acceptance criteria.
   - Require tests-only edits.
   - Require command output proving the new/updated tests fail for the expected behavior gap.
   - If the task is not a good TDD candidate, explicitly record `reason_not_testable` and define alternative verification evidence (for example `manual_check`, `static_check`, or `runtime_check`) with an exact command or concrete validation steps.
4. Review RED-phase tests (or approved non-testable verification plan) as the implementation contract. Do not weaken or remove tests unless requirements changed.
5. Implement production changes for all acceptance criteria.
6. Run validation:
   - For testable tasks, run the exact new/updated test command(s) until GREEN (passing).
   - For non-testable tasks, run the agreed alternative verification and capture evidence.
   - Run any additional validation steps from the plan if feasible.
7. Commit your work.
   - Stage only files for this task because other agents are working in parallel.
   - NEVER PUSH. ONLY COMMIT.
8. After the commit, update the `*-plan.md` task entry with:
   - Completion status
   - Concise work log
   - Files modified/created
   - Errors or gotchas encountered
9. Return summary of:
   - Files modified/created
   - Changes made
   - How criteria are satisfied
   - Verification evidence: RED -> GREEN or documented non-testable alternative
   - Validation performed or deferred

## Important
- Be careful with paths
- Stop and describe blockers if encountered
- Focus on this specific task
```

Ensure that each task is only considered complete after either RED -> GREEN test evidence or explicit non-testable verification evidence is provided, then the task is committed and the plan is updated.

### Step 4: Monitor & Collect Results

**For standard tasks (Task tool):** Results return directly from the subagent.

**For design tasks (`claude -p`):**
1. Check if background processes are still running: `ps -p [PID]`
2. When complete, read the output log: Read `/tmp/co-design-[TASK_ID]-output.log`
3. Verify the design agent completed its work by checking:
   - Files were created/modified as expected
   - Plan file was updated with task status
   - Work was committed

Wait for ALL tasks in the current wave to complete before proceeding.

### Step 5: Check and Validate

After all tasks in a wave complete:
1. Inspect their outputs for correctness and completeness.
2. Validate the results against the expected outcomes.
3. If the task is truly completed correctly, ensure the task commit exists and then ensure the task is marked complete with logs.
4. If a task was not successful, have the agent retry or escalate the issue.
5. Ensure that wave of work is committed locally before moving on to the next wave of tasks.

### Step 6: Repeat

1. Review the plan again to see what new set of unblocked tasks are available.
2. Continue launching unblocked tasks in parallel until plan is done.
3. Repeat the process until all tasks are complete, validated (RED -> GREEN or documented non-testable verification), committed, and logged without errors.

## Monitoring Design Agents

While waiting for design agents to complete, you can:

```bash
# Check if agent is still running
ps -p [PID] > /dev/null 2>&1 && echo "Running" || echo "Done"

# Tail the output log for progress
tail -20 /tmp/co-design-[TASK_ID]-output.log

# Check all running design agents
ps aux | grep "claude -p" | grep -v grep
```

## Error Handling

- Task subset not found: List available task IDs
- Parse failure: Show what was tried, ask for clarification
- Design agent crash: Read the output log, retry the task
- Design agent timeout: Check log for blockers, retry or escalate

## Example Usage

```
'Implement the plan using co-design skill'
/co-design plan.md
/co-design ./plans/dashboard-plan.md T1 T2 T4
/co-design landing-page-plan.md --tasks T3 T7
```

## Execution Summary Template

```markdown
# Execution Summary

## Tasks Assigned: [N]
- Design tasks (claude -p): [count]
- Standard tasks (subagent): [count]

### Completed
- Task [ID]: [Name] - [Brief summary] [🎨 design | ⚙️ standard]

### Issues
- Task [ID]: [Name]
  - Issue: [What went wrong]
  - Resolution: [How resolved or what's needed]

### Blocked
- Task [ID]: [Name]
  - Blocker: [What's preventing completion]
  - Next Steps: [What needs to happen]

## Overall Status
[Completion summary]

## Files Modified
[List of changed files]

## Next Steps
[Recommendations]
```
