"""Reporting which tools produced a reply.

Users want to know whether an answer came from a live tool or from the model's
own knowledge, so every tool annotates itself with a label and the arguments
worth showing, and every successful call becomes a :class:`ToolCall` which the
client renders as one line per call:

    🔧 Tool use
    • web search: query="time in tokyo"
    • read page: url="https://example.com/time"

Memory loading and saving are graph plumbing rather than tools: they never
produce a ``ToolCall``, so they never show up in the report.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

# where a tool's report annotation lives, inside its langchain ``metadata``
ANNOTATION_KEY: str = "synthea_report"

# rendering limits, chosen so a whole report fits in an embed field
MAX_VALUE_CHARS: int = 100
MAX_REPORT_CHARS: int = 1024  # discord's embed field value limit

TOOL_USE_FIELD_NAME: str = "🔧 Tool use"
NO_TOOLS_USED_TEXT: str = (
    "None — answered from the model's own knowledge, not live search."
)


@dataclass(frozen=True)
class ToolCall:
    """One successful tool call: the tool's real name (``tavily_search``), the
    label users see (``web search``) and its rendered arguments, such as
    ``query="time in tokyo"``.
    """

    name: str
    label: str
    detail: str = ""


def reported_as(tool: Any = None, *, label: str, show: Iterable[str] = ()) -> Any:
    """Annotates a tool with how its calls should be reported.

    Call it directly for a tool you only have an instance of, or as a decorator
    above ``@tool``::

        search_tool = reported_as(search_tool, label="web search", show=("query",))

        @reported_as(label="read page", show=("url", "query"))
        @tool
        async def read_url(url: str) -> str: ...

    ``show`` lists the arguments worth printing, in order; the ones the model
    did not pass are skipped. The annotation is kept in the tool's langchain
    ``metadata``, so it travels with the tool and needs no registry.
    """

    def annotate(target: Any) -> Any:
        metadata = dict(getattr(target, "metadata", None) or {})
        metadata[ANNOTATION_KEY] = (label, tuple(show))
        target.metadata = metadata
        return target

    return annotate(tool) if tool is not None else annotate


def describe_tool_call(tool: Any, args: Mapping[str, Any]) -> ToolCall:
    """Builds the report record for one successful tool call.

    A tool without an annotation still gets reported: it falls back to its raw
    name and to every argument it was called with.
    """
    name = getattr(tool, "name", None) or type(tool).__name__
    metadata = getattr(tool, "metadata", None)
    annotation = metadata.get(ANNOTATION_KEY) if isinstance(metadata, dict) else None
    label, show = annotation if annotation else (name, tuple(args))

    parts: list[str] = []
    for key in show:
        value = args.get(key)
        if value is None or (isinstance(value, (list, tuple)) and not value):
            continue  # an optional argument the model didn't pass
        text = " ".join(str(value).split())  # never let a value break the layout
        if len(text) > MAX_VALUE_CHARS:
            text = text[:MAX_VALUE_CHARS].rstrip() + "…"
        parts.append(f'{key}="{text}"' if isinstance(value, str) else f"{key}={text}")
    return ToolCall(name=name, label=label, detail=", ".join(parts))


def format_tool_use(tools_used: list[ToolCall] | None) -> str | None:
    """Formats the report for the user: one line per tool call.

    Returns None when tool use was not tracked for the response (nothing
    honest to report), and NO_TOOLS_USED_TEXT when no tools were called at all.
    """
    if tools_used is None:
        return None
    if not tools_used:
        return NO_TOOLS_USED_TEXT

    lines = [
        f"• {call.label}: {call.detail}" if call.detail else f"• {call.label}"
        for call in tools_used
    ]
    for end in range(len(lines), 0, -1):  # drop whole lines until it fits
        text = "\n".join(lines[:end])
        if end < len(lines):
            text += f"\n… (+{len(lines) - end} more)"
        if len(text) <= MAX_REPORT_CHARS:
            return text
    return lines[0][:MAX_REPORT_CHARS]
