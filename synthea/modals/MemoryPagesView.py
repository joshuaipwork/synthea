# -*- coding: utf-8 -*-
"""
A paginated view for browsing a user's stored memories.
"""
import discord
from discord import ButtonStyle, ui

MAX_PAGE_BUTTONS = 5
MAX_MEMORY_CHARS = 150
MAX_MESSAGE_CHARS = 1900
DEFAULT_PAGE_SIZE = 10


class MemoryPagesView(ui.View):
    """
    A view that paginates a (potentially large) list of memories so the user can
    browse them with previous/next buttons instead of a single oversized message.
    """

    def __init__(self, memories: list[dict[str, str]]):
        super().__init__(timeout=300)
        self.memories: list[dict[str, str]] = memories
        self.page_size: int = DEFAULT_PAGE_SIZE
        self.page: int = 0
        self.num_pages: int = max(1, -(-len(memories) // self.page_size))

        self.previous_button = ui.Button(label="<", style=ButtonStyle.blurple)
        self.next_button = ui.Button(label=">", style=ButtonStyle.blurple)
        self.previous_button.callback = self.go_to_previous_page
        self.next_button.callback = self.go_to_next_page

        self._update_buttons()
        self.add_item(self.previous_button)
        self.add_item(self.next_button)

    def _update_buttons(self):
        """Disables the navigation buttons at the start/end of the list."""
        self.previous_button.disabled = self.page == 0
        self.next_button.disabled = self.page >= self.num_pages - 1

    def _build_content(self) -> str:
        """Builds the message content for the current page."""
        start = self.page * self.page_size
        end = min(start + self.page_size, len(self.memories))
        page_memories = self.memories[start:end]

        lines = [f"Stored memories about you (page {self.page + 1}/{self.num_pages}):\n"]
        for m in page_memories:
            memory_text = str(m["memory"])
            if len(memory_text) > MAX_MEMORY_CHARS:
                memory_text = memory_text[: MAX_MEMORY_CHARS - 1].rstrip() + "..."
            lines.append(f"- {memory_text} ({m['id']})")

        content = "\n".join(lines)
        return content[:MAX_MESSAGE_CHARS]

    async def go_to_previous_page(self, interaction: discord.Interaction):
        """Moves to the previous page."""
        self.page = max(0, self.page - 1)
        self._update_buttons()
        await interaction.response.edit_message(content=self._build_content(), view=self)

    async def go_to_next_page(self, interaction: discord.Interaction):
        """Moves to the next page."""
        self.page = min(self.num_pages - 1, self.page + 1)
        self._update_buttons()
        await interaction.response.edit_message(content=self._build_content(), view=self)

    async def on_timeout(self) -> None:
        """On timeout, disable the buttons."""
        self.previous_button.disabled = True
        self.next_button.disabled = True
        return await super().on_timeout()
