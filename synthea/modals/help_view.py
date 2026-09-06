"""A paginated, self-deleting view for showing the bot's help text."""

import discord
from discord import ButtonStyle, ui

# how long the help message stays around before it deletes itself, in seconds
HELP_TIMEOUT_SECONDS = 300


class HelpView(ui.View):
    """Paginates a help message with previous/next buttons.

    The message is ephemeral: once the buttons time out (or the user deletes
    it early via the 🗑️ reaction), the help message removes itself from the
    channel.
    """

    def __init__(
        self,
        pages: list[str],
        title: str,
        timeout: float = HELP_TIMEOUT_SECONDS,
    ):
        super().__init__(timeout=timeout)
        self.pages: list[str] = pages
        self.title: str = title
        self.page: int = 0
        self.num_pages: int = max(1, len(pages))

        # set by `send` once the message is posted, so the view can delete
        # the message when the buttons time out
        self.message: discord.Message | None = None

        self.previous_button = ui.Button(label="◀️", style=ButtonStyle.secondary)
        self.next_button = ui.Button(label="▶️", style=ButtonStyle.secondary)
        self.previous_button.callback = self.go_to_previous_page
        self.next_button.callback = self.go_to_next_page

        self._update_buttons()
        self.add_item(self.previous_button)
        self.add_item(self.next_button)

    @classmethod
    async def send(
        cls, message: discord.Message, pages: list[str], title: str,
    ) -> discord.Message:
        """Sends the paginated help message as a reply to the given message."""
        view: HelpView = cls(pages, title)
        help_message: discord.Message = await message.reply(
            embed=view.build_embed(), view=view, mention_author=True,
        )
        view.message = help_message
        await help_message.add_reaction("🗑️")
        return help_message

    def build_embed(self) -> discord.Embed:
        """Builds the embed for the current page."""
        embed = discord.Embed(title=self.title, description=self.pages[self.page])
        footer = f"Page {self.page + 1}/{self.num_pages}"
        if self.num_pages > 1:
            footer += " • use ◀️ ▶️ to flip through"
        if self.timeout is not None:
            footer += f" • this message will be deleted in {max(1, round(self.timeout / 60))} minutes"
        embed.set_footer(text=footer)
        return embed

    def _update_buttons(self) -> None:
        """Disables the navigation buttons at the start/end of the list."""
        self.previous_button.disabled = self.page == 0
        self.next_button.disabled = self.page >= self.num_pages - 1

    async def go_to_previous_page(self, interaction: discord.Interaction) -> None:
        """Moves to the previous page."""
        self.page = max(0, self.page - 1)
        self._update_buttons()
        await interaction.response.edit_message(
            embed=self.build_embed(), view=self,
        )

    async def go_to_next_page(self, interaction: discord.Interaction) -> None:
        """Moves to the next page."""
        self.page = min(self.num_pages - 1, self.page + 1)
        self._update_buttons()
        await interaction.response.edit_message(
            embed=self.build_embed(), view=self,
        )

    async def on_timeout(self) -> None:
        """When the buttons time out, delete the help message (best effort)."""
        self.previous_button.disabled = True
        self.next_button.disabled = True
        if self.message is not None:
            try:
                await self.message.delete()
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                # the message may have been deleted already (e.g. via 🗑️)
                pass
        return await super().on_timeout()
