# ADR-001: LangGraph as the Agent Orchestration Framework

**Status:** Accepted  
**Date:** 2024-Q4  
**Deciders:** NorthSea AgentOps architecture team  

---

## Context

The investigation pipeline requires a stateful, multi-step agent graph where:
- Multiple agents (Planner, Executor, Critic, Challenger) hand off state
- A human-in-the-loop interrupt point is needed before the UncertaintyGate
- The executor must loop until all plan steps are complete (conditional routing)
- State must be checkpointed for resumability (human review may take minutes)
- Two distinct patterns must coexist: Plan-Execute-Critic AND ReAct

The framework must handle **durable state**, **conditional routing**, and **human checkpoints** out of the box — not as afterthoughts.

---

## Decision

Use **LangGraph** (LangChain Inc., 2024) as the primary agent orchestration framework.

---

## Options Considered

| Framework | Verdict | Reasons |
|-----------|---------|---------|
| **LangGraph** | ✅ CHOSEN | Explicit graph, native human-in-the-loop via `interrupt_before`, MemorySaver checkpointing, conditional edges, `astream()` for async streaming |
| **AutoGen** (Microsoft) | ❌ Rejected | Conversation-based multi-agent model (agents talk to each other). Poor fit for plan-execute loops. No native interrupt_before semantics. Weaker state persistence. |
| **CrewAI** | ❌ Rejected | Role-based crew abstraction is higher-level but loses control over routing logic. No built-in checkpointing for human-in-loop. Less suitable for offline/controlled industrial environments. |
| **LangChain AgentExecutor** | ❌ Rejected | Flat ReAct executor — can't express the Planner→Executor(loop)→Critic→Gate graph. No interrupt_before. Cannot do Challenger as a separate node. |
| **Custom asyncio graph** | ❌ Rejected | Would reinvent checkpointing, conditional routing, and streaming. Maintenance cost too high. |
| **Haystack Pipelines** | ❌ Rejected | Document pipeline focus, not general agentic loops. Human-in-loop not first-class. |

---

## Consequences

### Positive
- `interrupt_before=["uncertainty_gate"]` gives us native human review checkpoint with zero custom code
- MemorySaver enables resumable investigations — human can review, then `astream(None, config)` resumes
- `StateGraph(AgentState)` makes all agent state transitions explicit and auditable
- Conditional edges (`_should_continue_executing`, `_escalate_or_output`) are pure functions — testable without LLM
- `astream()` gives per-node streaming output for real-time progress in the API

### Negative / Trade-offs
- LangGraph is relatively new (2024); API stability is not at 1.0
- MemorySaver is in-process only — production needs PostgreSQL checkpointer (planned with `langgraph-checkpoint-postgres`)
- Learning curve is higher than AgentExecutor for developers new to graph-based agents

### Mitigations
- Pin LangGraph to `>=0.2.0` with upper bound during active development
- Add integration tests for the graph topology (node count, edge routing) independent of LLM
