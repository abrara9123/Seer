#!/usr/bin/env python3
"""Check and optionally package public source; private/runtime files are excluded."""
import argparse
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIRS = ("src", "include", "python", "tests", "tools", "assets", "models",
               "config", "docs", "requirements")
PUBLIC_FILES = ("README.md", "CMakeLists.txt", ".gitignore", ".clang-format")
SOURCE_SUFFIXES = {".cpp", ".hpp", ".mm", ".py", ".txt", ".conf", ".md", ".json", ".names"}


def public_files():
    paths = [ROOT / name for name in PUBLIC_FILES]
    for directory in PUBLIC_DIRS:
        paths.extend(path for path in (ROOT / directory).rglob("*")
                     if path.is_file() and path.suffix in SOURCE_SUFFIXES
                     and "__pycache__" not in path.parts)
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise RuntimeError(f"Unexpected public-source link: {path.name}")
    return sorted(paths)


def private_values():
    values = []
    key = ROOT / ".local/openai_api_key"
    if key.exists():
        values.append(key.read_text().strip())
    session = ROOT / ".local/chatgpt/sessions.json"
    if session.exists():
        def visit(value):
            if isinstance(value, dict):
                for name, item in value.items():
                    if name in {"access_token", "refresh_token", "id_token"} and isinstance(item, str):
                        values.append(item)
                    else:
                        visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(json.loads(session.read_text()))
    return [value for value in values if len(value) >= 20]


def check(paths):
    secrets = private_values()
    credential = re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b")
    for path in paths:
        content = path.read_text()
        if credential.search(content) or any(secret in content for secret in secrets):
            # Never include the matched secret in diagnostics.
            raise RuntimeError(f"Credential found in public file: {path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write a clean source ZIP after the checks pass.")
    args = parser.parse_args()
    paths = public_files()
    check(paths)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, Path("VisionAssist") / path.relative_to(ROOT))
        print(f"Created {args.output} with {len(paths)} public source files.")
    else:
        print(f"PASS: {len(paths)} public source files; no private credentials.")


if __name__ == "__main__":
    main()
