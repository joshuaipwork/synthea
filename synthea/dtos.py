from synthea.tool_report import ToolCall


class GenerationRequest:
    """

    """

    def __init__(self, response_index: int, context: str = "") -> None:
        # the response to update
        self.response_index: int = response_index
        self.context: str = context


class GenerationResponse:
    """An object representing the output of an LLM.
    """

    def __init__(
        self,
        final_output: str = "",
        reasoning: str = "",
        images: list[bytes] | None = None,
        tools_used: list[ToolCall] | None = None,
    ) -> None:
        # the response to update
        self.final_output: str = final_output
        self.reasoning: str = reasoning
        self.images: list[bytes] = images if images is not None else []
        # The tool calls made to produce this output, in call order: each one
        # carries the tool's label and the arguments it was called with (the
        # search query, the url, ...). Memory loading/saving is deliberately
        # not part of this list: only real tool calls are reported.
        # None means tool use was not tracked for this response (so nothing is
        # reported to the user), while an empty list means the agent ran and
        # called no tools at all (the reply came from the model's knowledge).
        self.tools_used: list[ToolCall] | None = tools_used


class ResponseUpdate:
    """ """

    def __init__(
        self,
        response_index: str,
        message_is_completed: bool,
        new_message: str = "",
        error: Exception = None,
    ) -> None:
        # the response to update
        self.response_index: int = response_index
        self.message_is_completed: bool = message_is_completed
        self.new_message: str = new_message
        self.error: Exception = error
