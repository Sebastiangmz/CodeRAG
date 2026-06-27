# Eval rubrics

Milestone 0 uses deterministic EvalFly assertions only.

Human or LLM judge rubrics are intentionally absent from the blocking baseline. Add them only after a deterministic check cannot express the risk and the acceptance contract names the reviewer, evidence path, pass conditions, fail conditions, and privacy classification.

Raw traces do not belong in this tree. Keep raw local material under `.pi/evalfly/raw/` and commit only sanitized fixtures under `evals/traces/sanitized/` when they protect an active eval case.
