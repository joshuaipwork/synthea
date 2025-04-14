class GenerationResponse:
    """
    An object representing the output of an LLM.
    """
    def __init__(self, final_output: str = "", reasoning: str = "", images: list[bytes] | None = None) -> None:
        # the response to update
        self.final_output: str = final_output
        self.reasoning: str = reasoning
        self.images: list[bytes] = images if images is not None else []