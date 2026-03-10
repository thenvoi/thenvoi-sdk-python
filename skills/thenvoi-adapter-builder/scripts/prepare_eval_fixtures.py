#!/usr/bin/env python3
"""Prepare eval fixtures by creating stripped copies of the repo.

Each fixture removes a single framework integration (adapter, converter, tests,
examples, exports, conformance wiring, and pyproject.toml optional deps) so the
skill can be evaluated on its ability to rebuild it from scratch.

Usage:
    python scripts/prepare_eval_fixtures.py [--frameworks gemini crewai letta]
    python scripts/prepare_eval_fixtures.py --all
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

SKILL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_DIR.parent.parent
FIXTURES_DIR = SKILL_DIR / "evals" / "fixtures"

# Directories and file patterns to always exclude when copying the repo.
COPY_EXCLUDES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    ".env",
    "node_modules",
    "skills",
    "dist",
    "*.egg-info",
}

# ---------------------------------------------------------------------------
# Framework stripping rules
# ---------------------------------------------------------------------------

# Files to delete outright for each framework.
FRAMEWORK_FILES: dict[str, list[str]] = {
    "crewai": [
        "src/thenvoi/adapters/crewai.py",
        "src/thenvoi/converters/crewai.py",
        "tests/adapters/test_crewai_adapter.py",
        "tests/converters/test_crewai.py",
        "examples/crewai",
    ],
    "letta": [
        "src/thenvoi/adapters/letta.py",
        "src/thenvoi/converters/letta.py",
        "tests/adapters/test_letta_adapter.py",
        "tests/converters/test_letta.py",
        "tests/integration/test_letta_live.py",
        "examples/letta",
    ],
    "gemini": [
        "src/thenvoi/adapters/gemini.py",
        "src/thenvoi/converters/gemini.py",
        "tests/adapters/test_gemini_adapter.py",
        "tests/converters/test_gemini.py",
        "examples/gemini",
    ],
}


def _should_exclude(path: Path) -> bool:
    """Return True if path matches any copy-exclude pattern."""
    for part in path.parts:
        if part in COPY_EXCLUDES:
            return True
        for pattern in COPY_EXCLUDES:
            if "*" in pattern and Path(part).match(pattern):
                return True
    return False


def copy_repo(dest: Path) -> None:
    """Copy the repo to dest, excluding non-essential directories."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for item in REPO_ROOT.iterdir():
        rel = item.relative_to(REPO_ROOT)
        if _should_exclude(rel):
            continue
        target = dest / rel
        if item.is_dir():
            shutil.copytree(
                item,
                target,
                ignore=shutil.ignore_patterns(*COPY_EXCLUDES),
            )
        else:
            shutil.copy2(item, target)


def delete_framework_files(dest: Path, framework: str) -> None:
    """Delete framework-specific files from the copied repo."""
    for rel_path in FRAMEWORK_FILES.get(framework, []):
        target = dest / rel_path
        if target.is_dir():
            shutil.rmtree(target)
            logger.info("Deleted directory: %s", rel_path)
        elif target.is_file():
            target.unlink()
            logger.info("Deleted file: %s", rel_path)
        else:
            logger.warning("File not found (skipping): %s", rel_path)


# ---------------------------------------------------------------------------
# Shared-file stripping: __init__.py exports
# ---------------------------------------------------------------------------


def _strip_adapter_init(dest: Path, framework: str) -> None:
    """Remove framework exports from adapters/__init__.py."""
    init_path = dest / "src" / "thenvoi" / "adapters" / "__init__.py"
    if not init_path.exists():
        return

    text = init_path.read_text()
    original = text

    # Class names to remove by framework
    class_map: dict[str, list[str]] = {
        "crewai": ["CrewAIAdapter"],
        "letta": ["LettaAdapter"],
        "gemini": ["GeminiAdapter"],
    }
    classes = class_map.get(framework, [])
    module = framework

    for cls_name in classes:
        # Remove TYPE_CHECKING import line
        text = re.sub(
            rf"^\s*from thenvoi\.adapters\.{module} import {cls_name} as {cls_name}\n",
            "",
            text,
            flags=re.MULTILINE,
        )
        # Remove from __all__
        text = re.sub(
            rf'^\s*"{cls_name}",?\n',
            "",
            text,
            flags=re.MULTILINE,
        )
        # Remove lazy import elif block (match full block to next elif/raise)
        text = re.sub(
            rf'    elif name == "{cls_name}":.*?(?=\n    elif |\n    raise )',
            "",
            text,
            flags=re.DOTALL,
        )

    # Remove install hint line
    text = re.sub(
        rf"^\s*uv add thenvoi-sdk\[{module}\]\n",
        "",
        text,
        flags=re.MULTILINE,
    )

    if text != original:
        init_path.write_text(text)
        logger.info("Stripped %s from adapters/__init__.py", framework)


def _strip_converter_init(dest: Path, framework: str) -> None:
    """Remove framework exports from converters/__init__.py."""
    init_path = dest / "src" / "thenvoi" / "converters" / "__init__.py"
    if not init_path.exists():
        return

    text = init_path.read_text()
    original = text

    # Class names to remove by framework
    class_map: dict[str, list[str]] = {
        "crewai": ["CrewAIHistoryConverter", "CrewAIMessages"],
        "letta": ["LettaHistoryConverter"],
        "gemini": ["GeminiHistoryConverter", "GeminiMessages"],
    }
    module = framework
    classes = class_map.get(framework, [])

    # Remove TYPE_CHECKING import block (multi-line from ... import (...))
    text = re.sub(
        rf"^\s*from thenvoi\.converters\.{module} import \(\n"
        rf"(.*?\n)*?\s*\)\n",
        "",
        text,
        flags=re.MULTILINE,
    )
    # Single-line TYPE_CHECKING import
    for cls_name in classes:
        text = re.sub(
            rf"^\s*from thenvoi\.converters\.{module} import\s+{cls_name}.*\n",
            "",
            text,
            flags=re.MULTILINE,
        )

    # Remove from __all__
    for cls_name in classes:
        text = re.sub(
            rf'^\s*"{cls_name}",?\n',
            "",
            text,
            flags=re.MULTILINE,
        )

    # Remove lazy import elif blocks - match the full block from elif to the line before the next elif/raise
    # Multi-class pattern: elif name in ("Cls1", "Cls2"): ... from ... import (...) ... return
    if len(classes) > 1:
        names_pattern = ", ".join(rf'"{c}"' for c in classes)
        text = re.sub(
            rf"    elif name in \({names_pattern}\):.*?(?=\n    elif |\n    raise )",
            "",
            text,
            flags=re.DOTALL,
        )
    # Single-class pattern: elif name == "Cls": ... return Cls
    for cls_name in classes:
        text = re.sub(
            rf'    elif name == "{cls_name}":.*?(?=\n    elif |\n    raise )',
            "",
            text,
            flags=re.DOTALL,
        )

    # Remove install hint line
    text = re.sub(
        rf"^\s*uv add thenvoi-sdk\[{module}\]\n",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Clean up excess blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    if text != original:
        init_path.write_text(text)
        logger.info("Stripped %s from converters/__init__.py", framework)


# ---------------------------------------------------------------------------
# Shared-file stripping: pyproject.toml
# ---------------------------------------------------------------------------


def _strip_pyproject(dest: Path, framework: str) -> None:
    """Remove the framework's optional dependency group from pyproject.toml."""
    pyproject = dest / "pyproject.toml"
    if not pyproject.exists():
        return

    lines = pyproject.read_text().splitlines(keepends=True)
    result: list[str] = []
    skip = False

    for line in lines:
        # Detect start of the target optional group
        stripped = line.strip()
        if stripped == f"{framework} = [":
            skip = True
            continue
        if skip:
            if stripped == "]":
                skip = False
            continue
        result.append(line)

    pyproject.write_text("".join(result))
    logger.info("Stripped [%s] optional deps from pyproject.toml", framework)


# ---------------------------------------------------------------------------
# Shared-file stripping: framework_configs
# ---------------------------------------------------------------------------


def _strip_adapter_config(dest: Path, framework: str) -> None:
    """Remove framework registration from tests/framework_configs/adapters.py."""
    config_path = dest / "tests" / "framework_configs" / "adapters.py"
    if not config_path.exists():
        return

    text = config_path.read_text()
    original = text

    # Remove the _build_<framework>_config entry from _ADAPTER_CONFIG_BUILDERS list
    text = re.sub(
        rf"^\s*_build_{framework}_config,?\n",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Remove the _build_<framework>_config function
    text = re.sub(
        rf"^def _build_{framework}_config\(\).*?(?=\ndef |\n# |\nADAPTER_EXCLUDED|\n_ADAPTER_CONFIG)",
        "",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )

    # Remove the factory function
    text = re.sub(
        rf"^def _{framework}_factory\(\*\*kw.*?(?=\ndef |\n# |\nclass )",
        "",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )

    # CrewAI has extra helpers: _MockBaseTool, _crewai_import_lock, etc.
    if framework == "crewai":
        # Remove _MockBaseTool class
        text = re.sub(
            r"^class _MockBaseTool:.*?(?=\n\n[a-zA-Z_])",
            "",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
        # Remove _crewai_import_lock
        text = re.sub(
            r"^_crewai_import_lock = threading\.Lock\(\)\n\n",
            "",
            text,
            flags=re.MULTILINE,
        )
        # Remove _CREWAI_AFFECTED_MODULES
        text = re.sub(
            r"^_CREWAI_AFFECTED_MODULES = \(.*?\)\n\n",
            "",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
        # Remove _get_crewai_adapter_cls
        text = re.sub(
            r"^@functools\.lru_cache.*?\ndef _get_crewai_adapter_cls\(\).*?(?=\n\n(async )?def )",
            "",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
        # Remove _crewai_conformance_guard
        text = re.sub(
            r"^async def _crewai_conformance_guard\(.*?(?=\ndef )",
            "",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
        # Remove conformance comment block
        text = re.sub(
            r"^# CrewAI conformance instances.*?(?=\nclass _MockBaseTool|\ndef _)",
            "",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )

    # Clean up excess blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    if text != original:
        config_path.write_text(text)
        logger.info("Stripped %s from framework_configs/adapters.py", framework)


def _strip_converter_config(dest: Path, framework: str) -> None:
    """Remove framework registration from tests/framework_configs/converters.py."""
    config_path = dest / "tests" / "framework_configs" / "converters.py"
    if not config_path.exists():
        return

    text = config_path.read_text()
    original = text

    # Remove from _CONVERTER_CONFIG_BUILDERS list
    text = re.sub(
        rf"^\s*_build_{framework}_config,?\n",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Remove the _build_<framework>_config function
    text = re.sub(
        rf"^def _build_{framework}_config\(\).*?(?=\ndef |\n# |\nCONVERTER_EXCLUDED|\n_CONVERTER_CONFIG)",
        "",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )

    # Remove the factory function
    text = re.sub(
        rf"^def _{framework}_factory\(\*\*kw.*?(?=\ndef |\n# )",
        "",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )

    # For letta: also remove from CONVERTER_EXCLUDED_MODULES
    if framework == "letta":
        text = re.sub(
            rf'^\s*"{framework}",?\n',
            "",
            text,
            flags=re.MULTILINE,
        )

    # Clean up excess blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    if text != original:
        config_path.write_text(text)
        logger.info("Stripped %s from framework_configs/converters.py", framework)


def _strip_output_adapters(dest: Path, framework: str) -> None:
    """Remove framework-specific output adapter if it's exclusive to this framework.

    SenderDictListAdapter is shared between CrewAI and Parlant, so it is NOT
    removed when stripping either one individually.
    """
    # Currently no output adapters are exclusive to a single framework that we strip.
    # This is a placeholder for future frameworks that might have exclusive adapters.
    pass


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def prepare_fixture(framework: str) -> Path:
    """Create a stripped fixture for a single framework."""
    fixture_name = f"{framework}-missing"
    dest = FIXTURES_DIR / fixture_name / "thenvoi-sdk-python"

    logger.info("Preparing fixture: %s -> %s", framework, dest)

    copy_repo(dest)
    delete_framework_files(dest, framework)
    _strip_adapter_init(dest, framework)
    _strip_converter_init(dest, framework)
    _strip_pyproject(dest, framework)
    _strip_adapter_config(dest, framework)
    _strip_converter_config(dest, framework)
    _strip_output_adapters(dest, framework)

    logger.info("Fixture ready: %s", fixture_name)
    return dest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Prepare eval fixtures")
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["crewai", "letta"],
        help="Frameworks to strip (default: crewai letta)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Prepare all fixtures (crewai, letta, gemini)",
    )
    args = parser.parse_args()

    frameworks = ["crewai", "letta", "gemini"] if args.all else args.frameworks

    for fw in frameworks:
        if fw not in FRAMEWORK_FILES:
            logger.error("Unknown framework: %s (known: %s)", fw, list(FRAMEWORK_FILES))
            sys.exit(1)

    for fw in frameworks:
        prepare_fixture(fw)

    logger.info("All fixtures prepared in %s", FIXTURES_DIR)


if __name__ == "__main__":
    main()
