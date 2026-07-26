#!/usr/bin/env python3
"""Generate API reference stubs and produce _zensical_build.toml.

Run before: uv run zensical build -f _zensical_build.toml --clean
For local development: uv run zensical serve -f _zensical_build.toml

Environment variables:
    VERSION_ALIAS  When set (e.g. "stable", "latest", "development"),
                   appends the alias to site_url in the output config so
                   that versioned builds have correct canonical URLs.
"""

import os
import re
import shutil
import tomllib
from pathlib import Path

# Module path prefixes (relative to src/) to exclude from the API docs.
# Use forward slashes, e.g. "mypkg/internal".
SKIP_PREFIXES: list[str] = [
    "ripplegw/benchmarks",
    "ripplegw/utils",
    "ripplegw/waveforms",
]
REFERENCE_TAB_NAME = "Reference"


# ── module scanning ───────────────────────────────────────────────────────────


def scan_modules(src_dir: Path) -> list[tuple[list[str], Path]]:
    """Return (module_parts, stub_path) for every public Python module."""
    results = []
    py_files = sorted(
        src_dir.rglob("*.py"),
        key=lambda p: tuple(
            part.casefold() for part in p.relative_to(src_dir).with_suffix("").parts
        ),
    )
    for py_file in py_files:
        parts = list(py_file.relative_to(src_dir).with_suffix("").parts)
        if parts[-1] == "__main__":
            continue
        if parts[-1] == "__init__":
            if len(parts) == 2:
                # The top-level package's own __init__.py becomes its Overview
                # page (e.g. "ripplegw"), so its __all__ is documented.
                parts = parts[:-1]
            else:
                continue
        if any(part.startswith("_") for part in parts):
            continue
        rel = "/".join(parts)
        if any(rel == skip or rel.startswith(skip + "/") for skip in SKIP_PREFIXES):
            continue
        stub_path = Path("api", *parts).with_suffix(".md")
        results.append((parts, stub_path))
    return results


def write_stubs(modules: list[tuple[list[str], Path]], docs_dir: Path) -> None:
    """Write (or overwrite) ::: identifier stubs under docs/api/."""
    api_dir = docs_dir / "api"
    keep = {docs_dir / stub_path for _, stub_path in modules}
    if api_dir.exists():
        for existing in api_dir.rglob("*.md"):
            if existing not in keep:
                existing.unlink()
    for parts, stub_path in modules:
        full_path = docs_dir / stub_path
        # Remove any conflicting stale entry left from a previous run
        assert api_dir in full_path.parents, f"stub path escapes api_dir: {full_path}"
        if full_path.exists() and full_path.is_dir():
            shutil.rmtree(full_path)
        if full_path.parent.exists() and full_path.parent.is_file():
            full_path.parent.unlink()
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(f"::: {'.'.join(parts)}\n", encoding="utf-8")
    # Prune empty directories left behind by removed stubs
    if api_dir.exists():
        for d in sorted(api_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if d.is_dir() and d != api_dir and not any(d.iterdir()):
                d.rmdir()


# ── waveform catalogue ─────────────────────────────────────────────────────────


def write_catalogue(docs_dir: Path) -> None:
    """Generate docs/guides/catalogue.md from the live waveform registry.

    Ripple-specific (not part of the shared template): every registered
    waveform's shape (domain, capabilities, constructor config,
    parameter_names) is introspected directly from the package rather than
    hand-maintained, so the catalogue cannot drift out of sync with the
    registry the way the old hand-written "supported waveforms" lists did.
    """
    import inspect

    import ripplegw
    from ripplegw.interfaces import AmplitudePhaseWaveform, DistanceScaledWaveform

    def format_config(cls) -> str:
        params = list(inspect.signature(cls.__init__).parameters.values())[1:]
        parts = [
            p.name if p.default is inspect.Parameter.empty else f"{p.name}={p.default!r}"
            for p in params
            if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]
        return ", ".join(parts) if parts else "(none)"

    def format_names(names) -> str:
        return ", ".join(f"`{n}`" for n in names)

    def yes_no_dash(value) -> str:
        if value is None:
            return "—"
        return "yes" if value else "no"

    rows = []
    details = []
    for name in ripplegw.list_waveforms():
        cls = ripplegw.WAVEFORM_REGISTRY[name]
        metadata = ripplegw.get_waveform_metadata(name)
        domain = metadata.get("domain", "?")
        dtype = {"FD": "complex", "TD": "real"}.get(domain, "?")
        has_amp_phase = issubclass(cls, AmplitudePhaseWaveform)
        has_distance = issubclass(cls, DistanceScaledWaveform)
        config = format_config(cls)
        ctor_params = inspect.signature(cls.__init__).parameters

        rows.append(
            f"| `{name}` | {domain} | {dtype} | {yes_no_dash(has_amp_phase)} | "
            f"{yes_no_dash(has_distance)} | {yes_no_dash(metadata.get('is_tidal'))} | "
            f"{yes_no_dash(metadata.get('is_precessing'))} | `{config}` |"
        )

        detail = [f"### `{name}`", ""]
        detail.append("Config: " + (f"`{config}`" if config != "(none)" else "no constructor arguments."))
        detail.append("")
        if "use_lambda_tildes" in ctor_params:
            detail.append(
                "`parameter_names` (`use_lambda_tildes=False`, default): "
                + format_names(cls(use_lambda_tildes=False).parameter_names)
            )
            detail.append("")
            detail.append(
                "`parameter_names` (`use_lambda_tildes=True`): "
                + format_names(cls(use_lambda_tildes=True).parameter_names)
            )
        else:
            detail.append("`parameter_names`: " + format_names(cls().parameter_names))
        detail.append("")
        details.append("\n".join(detail))

    assert rows, "catalogue generator produced no rows"

    lines = [
        "<!-- Generated by docs/gen_api.py — do not edit. -->",
        "",
        "# Waveform Catalogue",
        "",
        "Every waveform currently registered with `ripplegw.waveform(...)`. Construct any of them by "
        'name: `ripplegw.waveform("<Name>", **config)`. See [Working with Waveforms](waveforms.md) '
        "for the calling convention and [Parameters and Conventions](parameters.md) for what each "
        "parameter means.",
        "",
        "| Name | Domain | Output | `amplitude`/`phase` | `at_unit_distance` | Tidal | Precessing | Config |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
        "## Details",
        "",
        *details,
    ]

    out_path = docs_dir / "guides" / "catalogue.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── nav generation ────────────────────────────────────────────────────────────


def build_nav_tree(modules: list[tuple[list[str], Path]]) -> list:
    """Convert flat module list into a nested nav structure."""
    tree: dict = {}
    for parts, stub_path in modules:
        node = tree
        for part in parts[:-1]:
            existing = node.get(part)
            if isinstance(existing, str):
                node[part] = {"__init__": existing}
            node = node.setdefault(part, {})
        leaf = stub_path.as_posix()
        existing = node.get(parts[-1])
        if isinstance(existing, dict):
            existing["__init__"] = leaf
        else:
            node[parts[-1]] = leaf

    def dict_to_nav(d: dict) -> list:
        result = []
        for k in sorted(d, key=str.casefold):
            v = d[k]
            if isinstance(v, dict):
                children = []
                if "__init__" in v:
                    children.append({"Overview": v["__init__"]})
                children.extend(
                    dict_to_nav({ck: cv for ck, cv in v.items() if ck != "__init__"})
                )
                result.append({k: children})
            else:
                result.append({k: v})
        return result

    return dict_to_nav(tree)


def _toml_escape(s: str) -> str:
    """Escape backslashes and double-quotes for TOML basic strings."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def nav_to_toml_str(nav: list, depth: int = 0) -> str:
    """Serialize a nav list to the TOML array-of-inline-tables format."""
    inner = "    " * (depth + 1)
    lines = ["["]
    for item in nav:
        if isinstance(item, str):
            # Bare path entry (no label), e.g. "tutorials/index.md"
            lines.append(f'{inner}"{_toml_escape(item)}",')
        else:
            for k, v in item.items():
                if isinstance(v, str):
                    lines.append(
                        f'{inner}{{"{_toml_escape(k)}" = "{_toml_escape(v)}"}},'
                    )
                else:
                    nested = nav_to_toml_str(v, depth + 1)
                    lines.append(f'{inner}{{"{_toml_escape(k)}" = {nested}}},')
    lines.append("    " * depth + "]")
    return "\n".join(lines)


# ── TOML patching ─────────────────────────────────────────────────────────────


def replace_nav(toml_text: str, new_nav: list) -> str:
    """Replace the nav = [...] block in TOML text with a new nav list."""
    lines = toml_text.splitlines(keepends=True)
    start = end = None
    depth = 0
    for i, line in enumerate(lines):
        # NOTE: This regex only strips double-quoted strings on a single line.
        # It does not handle single-quoted strings, triple-quoted strings, or
        # backslash-escaped quotes. In practice zensical.toml uses only
        # double-quoted single-line strings in the nav block, so this is
        # sufficient, but a full TOML parser would be more robust.
        stripped = re.sub(r'"[^"]*"', "", line)
        if start is None and re.match(r"^\s*nav\s*=\s*\[", line):
            start = i
            depth = stripped.count("[") - stripped.count("]")
            if depth == 0:
                end = i
                break
        elif start is not None:
            depth += stripped.count("[") - stripped.count("]")
            if depth <= 0:
                end = i
                break
    if start is None:
        raise ValueError("nav key not found in zensical.toml")
    if end is None:
        raise ValueError("unterminated nav array in zensical.toml")
    new_block = "nav = " + nav_to_toml_str(new_nav) + "\n"
    return "".join([*lines[:start], new_block, *lines[end + 1 :]])


def patch_site_url(toml_text: str, new_url: str) -> str:
    """Replace site_url value in TOML text."""
    escaped_url = _toml_escape(new_url)
    new_text, count = re.subn(
        r'^(\s*site_url\s*=\s*")([^"]*)(")',
        lambda m: f"{m.group(1)}{escaped_url}{m.group(3)}",
        toml_text,
        count=1,
        flags=re.MULTILINE,
    )
    if count == 0:
        raise ValueError(
            "site_url not found in zensical.toml; cannot patch versioned URL"
        )
    return new_text


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    docs_dir = repo_root / "docs"
    src_dir = repo_root / "src"
    toml_path = repo_root / "zensical.toml"
    out_path = repo_root / "_zensical_build.toml"

    toml_text = toml_path.read_text(encoding="utf-8")
    config = tomllib.loads(toml_text)

    # Copy CONTRIBUTING.md so Zensical picks it up (symlinks and the name
    # "contributing.md" are both excluded by Zensical at build time)
    shutil.copy(repo_root / "CONTRIBUTING.md", docs_dir / "contributing.md")

    # Generate stubs and build API nav
    modules = scan_modules(src_dir)
    write_stubs(modules, docs_dir)
    api_nav = build_nav_tree(modules)
    print(f"Generated {len(modules)} API stubs")

    # Ripple-specific: regenerate the waveform catalogue from the live registry
    write_catalogue(docs_dir)
    print("Generated waveform catalogue")

    # Rebuild nav: keep everything except existing Reference entry, append new one
    def drop_reference_entries(nav):
        """Drop auto-generated reference sections so rebuilds stay idempotent."""
        return [
            item
            for item in nav
            if not (isinstance(item, dict) and REFERENCE_TAB_NAME in item)
        ]

    base_nav = drop_reference_entries(config.get("project", {}).get("nav", []))
    new_nav = [*base_nav, {REFERENCE_TAB_NAME: api_nav}]

    new_toml = replace_nav(toml_text, new_nav)

    # Optionally set versioned site_url
    alias = os.environ.get("VERSION_ALIAS", "").strip()
    if alias:
        site_url = config.get("project", {}).get("site_url")
        if site_url:
            base_url = site_url.rstrip("/") + "/"
            new_toml = patch_site_url(new_toml, f"{base_url}{alias}/")
        else:
            print(
                "Warning: VERSION_ALIAS set but site_url not found in zensical.toml; skipping URL patch"
            )

    out_path.write_text(new_toml, encoding="utf-8")
    print(f"Written {out_path}")


if __name__ == "__main__":
    main()
