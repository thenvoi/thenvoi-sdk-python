"""Entry point for ``python -m bridge_core``.

The bridge requires handlers to be registered before it can run.
Create your own entry point that registers handlers, for example::

    import asyncio
    from bridge_core.bridge import main
    from my_handlers import MyHandler

    asyncio.run(main(handlers={"my_handler": MyHandler()}))
"""

from __future__ import annotations


def _main() -> None:
    raise SystemExit(
        "The bridge requires handlers to be registered before it can run.\n"
        "\n"
        "Create your own entry point that registers handlers, for example:\n"
        "\n"
        "    import asyncio\n"
        "    from bridge_core.bridge import main\n"
        "    from my_handlers import MyHandler\n"
        "\n"
        "    asyncio.run(main(handlers={'my_handler': MyHandler()}))"
    )


if __name__ == "__main__":
    _main()
