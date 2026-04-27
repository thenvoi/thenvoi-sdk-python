# Predict Analysis - CrewAI Flow Adapter Spec

**Date:** 2026-04-27 09:46 PDT
**Scope:** Full area: CrewAI Flow spec, reason artifacts, current CrewAI adapter/runtime/converter surfaces, tests, conformance configs, and examples
**Personas:** 5 (Architecture Reviewer, Reliability Engineer, SDK DX Reviewer, Test Strategy Reviewer, Devil's Advocate)
**Debate Rounds:** 2
**Commit Hash:** `06b20f78045481f11ca8a150719c7201f856db21`
**Anti-Herd Status:** PASSED

## Summary

The spec is directionally right about the current CrewAI adapter failure: prompt/history inference is the wrong place to track pending peers, joins, sequential handoffs, and finalization. The reviewers also agreed that the current draft would still fail as a Genpact safety-net replacement unless it adds a real state hydration path, a stronger visible side-effect strategy, and a narrower Flow side-effect boundary.

Total findings: 10. Confirmed: 7. Probable: 3. Severity breakdown: Critical: 3, High: 5, Medium: 2, Low: 0.

## Top Findings

1. [Non-bootstrap peer replies will not see prior task events](./findings.md#finding-1) - CRITICAL | 5/5 consensus
2. [Visible sends are not idempotent or atomic with state events](./findings.md#finding-2) - CRITICAL | 5/5 consensus
3. [Raw runtime tools can bypass adapter safety policies](./findings.md#finding-3) - CRITICAL | 4/5 consensus
4. [Ambiguous reply matching should fail closed](./findings.md#finding-4) - HIGH | 5/5 consensus
5. [Task-event persistence conflicts with optional `AdapterFeatures.emit`](./findings.md#finding-5) - HIGH | 4/5 consensus

## What this means for the spec

The current plan can replace the model-history-inference part of the Genpact safety net. It cannot yet replace the operational safety net that prevents duplicate, premature, or stale visible output across retries, restarts, and non-bootstrap peer replies.

The minimum spec changes before implementation should be:

- add an explicit state hydration contract for every Flow turn, not just bootstrap history;
- change side-effect ordering to reserve/fence state before visible sends, or require platform idempotency keys;
- expose only policy-safe runtime helpers to Flow code;
- make ambiguous replies non-mutating unless a deterministic correlation token matches;
- make task-event persistence mandatory and independent of optional user-facing event emission.

## Files in This Report

- [Findings](./findings.md) - ranked failure modes and recommendations
- [Hypothesis Queue](./hypothesis-queue.md) - testable follow-up work
- [Persona Debates](./persona-debates.md) - condensed debate transcript
- [Codebase Analysis](./codebase-analysis.md)
- [Dependency Map](./dependency-map.md)
- [Component Clusters](./component-clusters.md)
- [Iteration Log](./predict-results.tsv)
