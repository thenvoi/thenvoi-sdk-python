---
id: TASK-005
phase: 4
status: completed
depends-on: [TASK-004]
---

# TASK-005: Add delegation, reply matching, and join handling

## Objective
Support delegation across multiple room messages. Implement the [Reply matching and idempotency rules](../technical-spec.md#reply-matching-and-idempotency-rules) verbatim.

## Spec reference
> **Goal:** Support delegation across multiple room messages. Implement the [Reply matching and idempotency rules](../technical-spec.md#reply-matching-and-idempotency-rules) verbatim.

## Changes
- [ ] `src/thenvoi/adapters/crewai_flow.py` — implement `delegate` handling: normalize each target via `normalize_participant_key`; reserve one `side_effect_key` per item; send one visible delegation message per confirmed reservation; record `delegated_pending`. When the same target has more than one pending delegation in the run, prepend the visible content with `[ref:{token}]` per the [Correlation token format](../technical-spec.md#correlation-token-format-v1).
- [ ] `src/thenvoi/adapters/crewai_flow.py` — implement reply matching exactly per the [rules](../technical-spec.md#reply-matching-and-idempotency-rules), including the `User`-vs-`Agent` sender filter on rule 8. A non-pending `Agent`-typed sender is discarded with a debug-level log; only `User`-typed senders can start a new run.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — implement `join_policy="all"` and `join_policy="first"`.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — implement `synthesize` handling with finalization checks and reserve-send-confirm.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — implement `indeterminate` recording when a reservation exists without sent confirmation on retry, applying the bounded confirmation-retry policy from [Task-event write requirements](../technical-spec.md#task-event-write-requirements-and-confirmation-retry).
- [ ] `tests/adapters/test_crewai_flow_adapter.py` — single-delegation, multi-delegation, first-reply join, all-replies join, duplicate delegation, duplicate finalization, ambiguous-reply fail-closed, token mismatch fail-closed, duplicate normalized-key fail-closed, User-typed unmatched reply, Agent-typed unmatched reply, confirmation-retry success, confirmation-retry-exhausted indeterminate, sent-confirm failure indeterminate.

## Acceptance criteria
- [ ] Unit: initial request with two delegate items records two reservations followed by two pending delegations.
- [ ] Unit: one reply under `join_policy="all"` records replied state and does not synthesize.
- [ ] Unit: second reply under `join_policy="all"` reserves and sends exactly one final response.
- [ ] Unit: one reply under `join_policy="first"` reserves and sends exactly one final response.
- [ ] Unit: ambiguous reply records `reply_ambiguous` and does not advance pending state.
- [ ] Unit: a User-typed sender with no matching pending delegation starts a new run.
- [ ] Unit: an Agent-typed sender with no matching pending delegation is discarded; no new run is created and existing pending state is unchanged.
- [ ] Unit: a retry whose reconstructed state shows a reservation without sent confirmation records `indeterminate` and sends no visible message; the test asserts both `len(tools.messages_sent)` is unchanged and a new task event with `status=indeterminate` is present.
- [ ] Unit: confirmation `send_event` fails 2 times then succeeds — the run records `delegated_pending` (or `finalized`) on the third attempt; visible message is not re-sent.
- [ ] Unit: confirmation `send_event` fails 3 times — the in-process result records `indeterminate`; on the next turn, reconstructed state shows reservation without sent record and the indeterminate path is followed; visible message is not re-sent.
- [ ] Pass criterion: `uv run pytest tests/adapters/test_crewai_flow_adapter.py -v -k "delegation or join or reply_matching or reservation or indeterminate or sender_filter or confirmation_retry"`
- [ ] Acceptance criterion 50 from spec: A `delegate` decision sends exactly one visible delegation message per item.
- [ ] Acceptance criterion 51: Each delegation records `run_id`, `delegation_id`, `target.normalized_key`, and `status=pending`.
- [ ] Acceptance criterion 52: Retrying the same `run_id` and `delegation_id` after a recorded `delegation_message_id` does not send a duplicate.
- [ ] Acceptance criterion 53: A sender matching exactly one pending delegation records `status=replied` and `reply_message_id`.
- [ ] Acceptance criterion 54: A sender matching multiple pending runs records `reply_ambiguous` and does not mutate delegation state.
- [ ] Acceptance criterion 55: A non-pending `User`-type sender starts a new run.
- [ ] Acceptance criterion 56: A non-pending `Agent`-type sender is discarded (debug log) and does not start a new run or mutate state.
- [ ] Acceptance criterion 57: `join_policy="all"` finalizes only after every pending delegation has `status=replied`.
- [ ] Acceptance criterion 58: `join_policy="first"` finalizes after the first matching reply.
- [ ] Acceptance criterion 59: A retry whose reconstructed state shows a reservation without a sent record records `status=indeterminate`, sends no visible message, and the next turn observes `len(tools.messages_sent)` unchanged.
- [ ] Acceptance criterion 60: A confirmation `send_event` failure is retried up to 3 times with backoff. If all 3 retries fail, the in-process result records `indeterminate`; the visible message is not re-sent on the next turn.
- [ ] Acceptance criterion 61: After a sent-confirm task-event failure that exhausts retry, the next on-message turn observes a reservation without sent confirmation and records `status=indeterminate`; no second visible message is sent for the same `side_effect_key`.
- [ ] Acceptance criterion 62: Token mismatch and duplicate normalized-key tests record `reply_ambiguous` and send no visible message.
- [ ] Acceptance criterion 63: `uv run pytest tests/adapters/test_crewai_flow_adapter.py -v -k "delegation or join or reply_matching or reservation or indeterminate or sender_filter"` passes.

## Out of scope
- Tagged-peer policy (Phase 5).
- Sequential chains (Phase 5).
- Buffered syntheses for partial finals (Phase 5).
- Public registration (Phase 6).
