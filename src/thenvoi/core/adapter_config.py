"""Shared helper for constructing adapters from typed config dataclasses."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import TypeVar

AdapterT = TypeVar("AdapterT")


def create_adapter_from_config(adapter_cls: type[AdapterT], config: object) -> AdapterT:
    """Instantiate an adapter class from a dataclass config object."""
    if not is_dataclass(config):
        raise TypeError(
            "config must be a dataclass instance for create_adapter_from_config()"
        )

    kwargs = asdict(config)
    if not isinstance(kwargs, dict):
        raise TypeError("dataclass config did not produce mapping kwargs")

    return adapter_cls(**kwargs)


__all__ = ["create_adapter_from_config"]
