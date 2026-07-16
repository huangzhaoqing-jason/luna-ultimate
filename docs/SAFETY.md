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
   → operator auth → action classification → audit. Every action passes one gate.
4. **Red-team test suite** (`safety/tests.py`) — exercised at boot and inside
   self-evolution fitness. A failing safety test blocks promotion.
5. **Tamper-evident audit log** (`safety/audit.py`) — hash-chained, append-only.
6. **Sandboxed automated programming** (`code_evolve/`) — proposes patches,
   safety-gates them, runs correctness + safety tests in a no-network subprocess,
   accepts only if both pass, reverts otherwise. `safety/` is always refused.

## What "loyalty" means here

The system is loyal to the operator's **legitimate intent**:
- it prioritizes the operator's goals over other users',
- it protects the operator's privacy and PII,
- it keeps the operator's confidences in the audit log,

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
python scripts/run_safety_tests.py --strict
```

Exits non-zero if any red-team case is misclassified, or if the charter hash
or audit chain is invalid.

## Automated programming self-evolution

```bash
python -m code_evolve.loop --max_iterations 3 --max_proposals 3
```

Each candidate patch is:
1. refused outright if it touches `safety/`,
2. gated by `SafetyLock.gate_code_patch` (charter + classifier + audit),
3. applied to disk,
4. run through `safety.tests.SafetyTestSuite` + a correctness command in a
   sandboxed subprocess with scrubbed env and hard timeout,
5. accepted only if both return 0; otherwise reverted.

## Charter text

See `safety/charter.py::SAFETY_CHARTER`. It is intentionally short and
immutable. To change it, a human must edit the source AND re-run
`enroll`/`verify_install`; the install-time hash record will flag the change.

## Limits / operator responsibilities

- True network isolation for generated code requires OS-level sandboxing
  (firejail, namespaces, seccomp). The loop scrubs env and disables common
  network libs but is not a hard network guarantee — configure the host.
- The classifier is conservative substring/regex; production deployments
  should plug in a stronger classifier at `safety.policy.classify_text`.
- The charter does not exempt the operator. Do not attempt to bypass it.
