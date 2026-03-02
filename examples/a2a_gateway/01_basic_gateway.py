"""Numbered alias entrypoint for examples/a2a_gateway/basic_gateway.py."""

from __future__ import annotations

import asyncio

from examples.a2a_gateway.basic_gateway import main


if __name__ == "__main__":
    asyncio.run(main())
