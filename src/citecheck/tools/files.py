"""Filesystem tools: `list_dir`, `read_file`, `save_report`.

Every path is resolved and containment-checked against a root before any I/O
happens (see `sandbox.py`). `read_file` roots at the workspace; `save_report`
roots at the outbox, so the agent can read broadly but write only where output
belongs.
"""

from __future__ import annotations

import hashlib

from citecheck.tools import sandbox
from citecheck.tools.base import Source, ToolContext, ToolResult


def list_dir(path: str, ctx: ToolContext) -> ToolResult:
    """List the contents of a directory inside the workspace."""
    s = ctx.settings

    def op() -> list[dict]:
        return sandbox.safe_list(path, s.workspace_dir)

    entries = sandbox.guarded(op)
    if isinstance(entries, ToolResult):
        return entries
    return ToolResult(data={"path": str(path), "entries": entries},
                      meta={"count": len(entries)})


def read_file(path: str, ctx: ToolContext) -> ToolResult:
    """Read a UTF-8 text file from the workspace and register it as a source."""
    s = ctx.settings
    holder: dict[str, str] = {}

    def op() -> str:
        text = sandbox.safe_read(path, s.workspace_dir, max_bytes=s.max_file_bytes)
        holder["text"] = text
        return text

    text = sandbox.guarded(op)
    if isinstance(text, ToolResult):
        return text

    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    name = str(path).replace("\\", "/")
    source = Source(
        id=f"F{digest[:8]}",
        uri=f"file://{name}",
        title=name.rsplit("/", 1)[-1],
        text=text[:20_000],
        kind="file",
        content_hash=digest,
    )
    return ToolResult(
        data={"source_id": source.id, "path": name, "chars": len(text),
              "text": source.text},
        source=source,
        meta={"bytes": len(text.encode("utf-8", errors="replace"))},
    )


def save_report(path: str, markdown: str, ctx: ToolContext) -> ToolResult:
    """Write a markdown report into the outbox. Confined by the sandbox."""
    s = ctx.settings
    if not (markdown or "").strip():
        return ToolResult.failure("empty report content")

    def op() -> str:
        written = sandbox.safe_write(
            path, s.outbox_dir, markdown, max_bytes=s.max_file_bytes
        )
        return str(written)

    written = sandbox.guarded(op)
    if isinstance(written, ToolResult):
        return written
    return ToolResult(
        data={"saved_to": written, "chars": len(markdown)},
        meta={"root": str(s.outbox_dir)},
    )


def read_workspace_tree(ctx: ToolContext, *, limit: int = 200) -> ToolResult:
    """Recursive listing of the workspace -- helps the agent find files in one call."""
    s = ctx.settings
    root = s.workspace_dir.resolve()
    entries: list[dict] = []
    for p in sorted(root.rglob("*")):
        if len(entries) >= limit:
            break
        entries.append({
            "path": str(p.relative_to(root)).replace("\\", "/"),
            "type": "dir" if p.is_dir() else "file",
            "size": p.stat().st_size if p.is_file() else None,
        })
    return ToolResult(data={"root": "workspace", "entries": entries},
                      meta={"count": len(entries)})