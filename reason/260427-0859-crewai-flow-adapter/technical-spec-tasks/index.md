# Tasks: CrewAI Flow Adapter

**Spec:** ../technical-spec.md
**PRD/BR:** none provided
**Created:** 2026-04-27
**Status:** 0/8 complete

## Task list

| ID | Title | Phase | Depends on | Status |
|----|-------|-------|------------|--------|
| TASK-001 | Extract shared CrewAI tool wrappers | 0 | — | pending |
| TASK-002 | Add the orchestration state contract and state source | 1 | — | pending |
| TASK-003 | Add the adapter API skeleton | 2 | — | pending |
| TASK-004 | Execute message-scoped Flows | 3 | TASK-001, TASK-002, TASK-003 | pending |
| TASK-005 | Add delegation, reply matching, and join handling | 4 | TASK-004 | pending |
| TASK-006 | Implement router safety policies | 5 | TASK-005 | pending |
| TASK-007 | Register the adapter surface | 6 | TASK-006 | pending |
| TASK-008 | Add the runnable example | 7 | TASK-007 | pending |

## Notes

- TASK-001, TASK-002, TASK-003 are independent and can be executed in parallel per the spec's dependency graph (`P0 --> P3`, `P1 --> P3`, `P2 --> P3`).
- TASK-004 cannot start until all three predecessors are complete.
- The remaining tasks are strictly sequential.
