"""Filesystem confinement.

The rule this module enforces: a path handed to any file tool is resolved, then
must live inside an allowed root. The check happens in Python after resolution,
so `../`, absolute paths, and Windows drive/UNC tricks all collapse to a single
real path before being compared. The model is never asked to behave -- the code
simply refuses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from citecheck.tools.base import ToolResult


class SandboxViolation(Exception):
    """A path escaped its allowed root."""


def resolve_in(root: Path, candidate: str, *, must_exist: bool = False) -> Path:
    """Resolve `candidate` against `root` and assert containment.

    Raises SandboxViolation. Returns the resolved absolute path on success.
    """
    root_resolved = root.resolve()
    raw = Path(candidate)

    # An absolute path is accepted but re-anchored against the root, so the
    # containment check below still governs it.
    target = (root_resolved / raw).resolve() if not raw.is_absolute() else raw.resolve()

    if not _is_within(target, root_resolved):
        raise SandboxViolation(
            f"path escapes sandbox: {candidate!r} resolves outside {root_resolved}"
        )

    if must_exist and not target.exists():
        raise FileNotFoundError(f"no such file: {candidate}")

    return target


def _is_within(target: Path, root: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


def safe_read(path: str | Path, root: Path, *, max_bytes: int) -> str:
    """Read a UTF-8 file after confirming it is a regular file inside `root`."""
    resolved = resolve_in(root, str(path), must_exist=True)
    if not resolved.is_file():
        raise SandboxViolation(f"not a regular file: {resolved.name}")
    size = resolved.stat().st_size
    if size > max_bytes:
        raise ValueError(f"file too large ({size} bytes > {max_bytes} limit)")
    # Errors="replace" so a stray binary or bad encoding cannot crash the run.
    return resolved.read_text(encoding="utf-8", errors="replace")


def safe_write(path: str | Path, root: Path, content: str, *, max_bytes: int) -> Path:
    """Write a UTF-8 file inside `root`, creating parent dirs as needed."""
    resolved = resolve_in(root, str(path))
    if len(content.encode("utf-8")) > max_bytes:
        raise ValueError("content exceeds max_bytes")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    # Containment was verified pre-resolution; re-verify after mkdir, which can
    # follow a symlinked parent.
    if not _is_within(resolved.resolve(), root.resolve()):
        raise SandboxViolation("write target escaped sandbox after symlink resolution")
    resolved.write_text(content, encoding="utf-8")
    return resolved


def safe_list(path: str | Path, root: Path) -> list[dict[str, Any]]:
    """List a directory inside `root`, one level deep."""
    resolved = resolve_in(root, str(path), must_exist=True)
    if not resolved.is_dir():
        raise NotADirectoryError(f"not a directory: {resolved.name}")
    out: list[dict[str, Any]] = []
    for child in sorted(resolved.iterdir(), key=lambda p: p.name):
        out.append(
            {
                "name": child.name,
                "type": "dir" if child.is_dir() else "file",
                "size": child.stat().st_size if child.is_file() else None,
            }
        )
        if len(out) >= 500:
            break
    return out


def guarded(fn, *args, **kwargs):
    """Run a sandboxed operation, converting expected failures into ToolResult."""
    try:
        return fn(*args, **kwargs)
    except SandboxViolation as e:
        return ToolResult.failure(f"sandbox_violation: {e}", blocked=True)
    except FileNotFoundError as e:
        return ToolResult.failure(f"not_found: {e}")
    except NotADirectoryError as e:
        return ToolResult.failure(f"not_a_directory: {e}")
    except PermissionError as e:
        return ToolResult.failure(f"permission_denied: {e}")
    except ValueError as e:
        return ToolResult.failure(f"invalid: {e}")
    except OSError as e:
        return ToolResult.failure(f"io_error: {e}")