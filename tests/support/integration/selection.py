"""Peer selection helpers for deterministic integration fixtures."""

from __future__ import annotations


def _peer_sort_key(peer: object) -> tuple[str, str, str]:
    handle = str(getattr(peer, "handle", "") or "")
    name = str(getattr(peer, "name", "") or "")
    peer_id = str(getattr(peer, "id", "") or "")
    return (handle, name, peer_id)


def _select_preferred_peer(
    peers: list[object],
    *,
    exclude_peer_id: str | None = None,
) -> object | None:
    """Pick a deterministic peer for integration fixtures."""
    if not peers:
        return None

    ordered = sorted(peers, key=_peer_sort_key)
    if exclude_peer_id is None:
        return ordered[0]

    for peer in ordered:
        if str(getattr(peer, "id", "")) != exclude_peer_id:
            return peer
    return ordered[0]


__all__ = ["_peer_sort_key", "_select_preferred_peer"]
