"""Peer identity directory shared by A2A gateway server and adapter."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from thenvoi_rest import Peer


@dataclass(frozen=True)
class PeerRef:
    """Canonical peer reference used across gateway boundaries."""

    slug: str
    peer: Peer


class PeerDirectory:
    """Centralized slug/UUID peer resolution."""

    def __init__(
        self,
        peers: Mapping[str, Peer] | None = None,
        peers_by_uuid: Mapping[str, Peer] | None = None,
    ) -> None:
        self._peers: dict[str, Peer] = {}
        self._peers_by_uuid: dict[str, Peer] = {}
        self._slug_by_uuid: dict[str, str] = {}
        self.load(peers=peers or {}, peers_by_uuid=peers_by_uuid or {})

    @property
    def peers(self) -> dict[str, Peer]:
        """Return peers keyed by canonical slug."""
        return self._peers

    @property
    def peers_by_uuid(self) -> dict[str, Peer]:
        """Return peers keyed by UUID aliases."""
        return self._peers_by_uuid

    def items(self) -> Iterable[tuple[str, Peer]]:
        """Iterate canonical slug → peer mappings."""
        return self._peers.items()

    def clear(self) -> None:
        """Clear all peer mappings."""
        self._peers.clear()
        self._peers_by_uuid.clear()
        self._slug_by_uuid.clear()

    def load(self, peers: Mapping[str, Peer], peers_by_uuid: Mapping[str, Peer]) -> None:
        """Replace directory contents from slug and UUID maps."""
        self.clear()

        for slug, peer in peers.items():
            self._register(slug=slug, peer=peer, uuid_alias=peer.id)

        for uuid_alias, peer in peers_by_uuid.items():
            slug = self._slug_by_uuid.get(peer.id) or self._slug_by_uuid.get(uuid_alias)
            if slug is None:
                slug = peer.handle or peer.id
            self._register(slug=slug, peer=peer, uuid_alias=uuid_alias)
            self._register(slug=slug, peer=peer, uuid_alias=peer.id)

    def replace_from_peers(
        self,
        peers: Iterable[Peer],
        *,
        slugify: Callable[[str], str],
    ) -> None:
        """Rebuild directory from a peer iterable."""
        self.clear()
        for peer in peers:
            slug = slugify(peer.name)
            self._register(slug=slug, peer=peer, uuid_alias=peer.id)

    def resolve(self, peer_id: str) -> PeerRef | None:
        """Resolve by slug first, then UUID aliases."""
        peer = self._peers.get(peer_id)
        if peer is not None:
            return PeerRef(slug=peer_id, peer=peer)

        peer = self._peers_by_uuid.get(peer_id)
        if peer is None:
            return None

        slug = self._slug_by_uuid.get(peer_id) or self._slug_by_uuid.get(peer.id)
        if slug is None:
            slug = peer.handle or peer.id
            self._register(slug=slug, peer=peer, uuid_alias=peer.id)
            self._slug_by_uuid[peer_id] = slug

        return PeerRef(slug=slug, peer=peer)

    def _register(self, *, slug: str, peer: Peer, uuid_alias: str) -> None:
        self._peers[slug] = peer
        self._peers_by_uuid[uuid_alias] = peer
        self._slug_by_uuid[uuid_alias] = slug
        self._slug_by_uuid[peer.id] = slug
