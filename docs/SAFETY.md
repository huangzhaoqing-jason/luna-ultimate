# Luna Safety & Operator-Loyalty Model

## Honest framing

"Absolute safety" and "absolute loyalty to one person" are **not achievable
guarantees** in any system. Luna approximates them with layered, defense-in-depth
engineering:

1. **Immutable safety charter** (`safety/charter.py`) — a hardcoded constant
   whose hash is verified at boot and on every gated action. No actor may
   modify it — not the operator, not self-evolution, not automated programming.
2. **Authorized operator as principal** (`safety/identity.py`) — the operator
   (黄照清, DOB 2013-05-07, CN) is recognized and prioritized for *legitimate*
   goals. Authorization is **not** permission to violate the charter.
3. **Layered safety locks** (`safety/locks.py`) — charter check → kill switch
   → operator auth → action classification → **CTM cognitive judge** → audit.
   The CTM judge can only ADD refusals on top of the charter; it never
   overturns a charter refusal.
4. **CTM+JEPA cognitive safety loop** (`safety/cognition.py`) — a dedicated
   `SafetyCTM` (reusing `modeling_ctm.CTM`) runs 1–4 adaptive ticks of
   neuron-level reasoning over each action, predicts the consequence via
   CTM-JEPA, and compares it against a frozen `ForbiddenPrototypeSet`.
   Untrained, it abstains (low confidence → allow) so it does not randomly
   block legitimate actions.
5. **Red-team + cognitive test suite** (`safety/tests.py`) — exercised at boot
   and inside self-evolution fitness. A failing safety or cognitive test
   blocks promotion.
6. **Tamper-evident audit log** (`safety/audit.py`) — hash-chained, append-only;
   each entry carries the CTM reasoning trace.
7. **Sandboxed automated programming** (`code_evolve/`) — proposes patches,
   safety-gates them, runs correctness + safety tests in a no-network subprocess,
   accepts only if both pass, reverts otherwise. `safety/` is always refused.

## Cognitive safety loop (the "thought process")

```
action_text + operator_token
  → charter hard floor (regex classify_text)        # immutable, non-negotiable
  → ActionEncoder + OperatorEmbedding + CharterEmbedding → hidden
  → SafetyCTM (CTM, 1-4 adaptive ticks)             # the thought process
  → CTMJEPAPredictor → predicted consequence
  → ForbiddenPrototypeSet cosine similarity          # "is the outcome forbidden?"
  → SafetyJudgment(allow/refuse, confidence, trace)
  → final = charter_refuse OR ctm_refuse             # CTM can only add refusals
  → AuditLog (with reasoning trace)
```

**Invariant**: `final_refuse = charter_refuse OR ctm_refuse`. The CTM can
refuse something the charter allowed (catching subtle harm the regex misses),
but it can **never** allow something the charter refused. The charter is the
hard floor; the CTM is the soft, learned layer on top.

When untrained, `SafetyCTM` returns low confidence and abstains, so it does
not randomly block legitimate operator goals. After training (future work),
it becomes the primary catcher of subtle, paraphrased, or novel harms.

## What "loyalty" means here

The system is loyal to the operator's **legitimate intent**:
- it prioritizes the operator's goals over other users',
- it protects the operator's privacy and PII,
- it keeps the operator's confidences in the audit log,
- the operator's verified identity is injected as a conditioning embedding
  (`OperatorEmbedding`) into both the main model and the SafetyCTM, so the
  system literally "recognizes" its operator.

It is **not** loyal in the sense of executing harmful commands from the
operator. The charter binds the operator exactly as it binds everyone else.
This is the only design under which "safety" and "loyalty" can coexist.

## Enrolling the operator

```bash
python scripts/enroll_operator.py
# → prints a ONE-TIME token; store it securely
```

Only a salted PBKDF2 hash of the token is stored in `.luna/operator.token`.
The operator's public claims (name, DOB, nationality, operator_id) are in
`safety/identity.py::AUTHORIZED_OPERATOR`.

## Running safety tests

```bash
python scripts/run_safety_tests.py --strict            # red-team suite
python scripts/run_safety_cognition.py --run-cognitive-tests  # CTM invariants
python scripts/run_safety_cognition.py --action "explain quicksort"
```

Exits non-zero if any red-team case is misclassified, any cognitive invariant
fails, or the charter hash / audit chain is invalid.

## Automated programming self-evolution

```bash
python -m code_evolve.loop --max_iterations 3 --max_proposals 3
```

Each candidate patch is:
1. refused outright if it touches `safety/`,
2. gated by `SafetyLock.gate_code_patch` (charter + CTM cognitive judge + audit),
3. applied to disk,
4. run through `safety.tests.SafetyTestSuite` (red-team + cognitive) + a
   correctness command in a sandboxed subprocess with scrubbed env and hard
   timeout,
5. accepted only if both return 0; otherwise reverted.

`code_evolve/patcher.py::PatchProposer` exposes a `proposer` hook so an
external LLM generator can be plugged in later (Q2B extension point); the
safety gate still applies to every proposal regardless of generator.

## Charter text

See `safety/charter.py::SAFETY_CHARTER`. It is intentionally short and
immutable. To change it, a human must edit the source AND re-run
`enroll`/`verify_install`; the install-time hash record will flag the change.

## Limits / operator responsibilities

- True network isolation for generated code requires OS-level sandboxing
  (firejail, namespaces, seccomp). The loop scrubs env and disables common
  network libs but is not a hard network guarantee — configure the host.
- The regex classifier is conservative; the CTM cognitive layer is the
  soft catcher for subtle/paraphrased harms but only becomes effective after
  training. Until then it abstains safely.
- The charter does not exempt the operator. Do not attempt to bypass it.
