# ADR-005: Multi-Agent Coordination Patterns

**Status:** Accepted  
**Date:** 2025-Q1  

---

## Context

Aker BP Track A explicitly requires:
- "Experience with multi-agent coordination patterns (planner-executor, critic-actor, etc.)"
- "Calibrated uncertainty estimation for agent decision points"
- "Design, evaluate, and harden agent workflows"

The system needs to handle two distinct investigation complexity tiers:
1. **Complex anomaly investigations** — multi-step, multi-tool, safety-critical, needs adversarial review
2. **Simple operational queries** — single-turn, fast, low-cost

Using the same architecture for both would be wasteful and would conflate different reliability requirements.

---

## Decision

Implement **two distinct agentic patterns** in the same codebase, sharing tools and safety infrastructure:

### Pattern 1: Plan-Execute-Critic-Challenger (LangGraph)
For complex anomaly investigations.

```
START → Planner → Executor(loop) → Critic → Challenger* → UncertaintyGate → (Escalate | Output) → END
```

- **Planner** (Plan-and-Execute, LangChain 2023): Upfront decomposition of investigation into 3-5 steps with tool assignments
- **Executor** (loop): Executes steps sequentially, dispatching tools per the plan
- **Critic** (Critic-Actor pattern): Independent validation pass — generates hypothesis, scores confidence, identifies gaps
- **Challenger** (Adversarial multi-agent): Devil's advocate — surfaces alternative hypotheses, challenges Critic's confidence. Activates only for HIGH/CRITICAL risk or confidence < 0.80.
- **Reconciler** (within Challenger): Synthesises Critic + Challenger into conservative final confidence
- **UncertaintyGate**: 5-rule HSE-aligned escalation decision

### Pattern 2: ReAct (Reasoning + Acting)
For single-turn operational Q&A.

```
Query → [Thought → Action → Observation]* → Final Answer
```

- Yao et al. (2022): https://arxiv.org/abs/2210.03629
- Interleaved reasoning and tool use — no upfront planning overhead
- Up to `MAX_REACT_STEPS = 6` cycles (safety cap)
- Shares the same tool allowlist and injection guards as Pattern 1

---

## Pattern Selection Criteria

| Query type | Pattern | Rationale |
|------------|---------|-----------|
| Anomaly alert with severity ≥ MEDIUM | Plan-Execute-Critic-Challenger | Multi-step evidence gathering needed; Critic+Challenger for safety |
| Single tool lookup ("BHP on F-4?") | ReAct | Single step; plan overhead unjustified |
| HSE procedure retrieval | ReAct | Single retrieve_documents call |
| Complex root cause with missing data | Plan-Execute-Critic-Challenger | Multiple evidence steps; Challenger ensures no overconfidence |

---

## Shared Infrastructure

Both patterns share:
- **Tool allowlist** (`src/safety/tool_allowlist.py`): Blocked/human-required tools enforced for both
- **Injection guard** (`src/safety/injection_guard.py`): User query + retrieved chunk sanitisation
- **Audit log** (`src/safety/audit_log.py`): All tool calls and escalation decisions logged
- **Prometheus metrics** (`src/observability/telemetry.py`): Investigation duration, confidence, escalation rate
- **OTEL tracing**: Both patterns emit spans to the collector

---

## Uncertainty Calibration Strategy

The Critic produces a confidence score in [0, 1]. The calibration strategy:

1. **Penalty for gaps**: If Critic identifies ≥3 logical gaps, confidence is reduced by 0.15 (hard rule)
2. **Challenger reconciliation**: If Challenger disagrees (PARTIAL/DISAGREE), confidence is conservatively reduced to `min(critic_conf, challenger_conf, 0.75)`
3. **Gate threshold**: Escalate if `confidence < 0.75` OR `risk_level in (HIGH, CRITICAL)` OR `evidence_coverage < 0.60`
4. **ECE measurement**: Offline calibration using `eval/calibration.py` with `CALIBRATION_VALIDATION_CASES`

Target: ECE ≤ 0.10 (GOOD grade), escalation recall ≥ 0.85.

---

## Consequences

- Codebase demonstrates three named agentic patterns: Plan-Execute, Critic-Actor, ReAct
- Multi-agent coordination is explicit (Challenger → Reconciler → state update)
- Confidence calibration is measurable in CI via `eval/agent_eval.py`
- Pattern selection is documented and testable
- Token cost is managed: Challenger is cost-gated, ReAct is used for simple queries
