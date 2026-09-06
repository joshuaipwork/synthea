"""Tests for the paginated ephemeral help view in synthea.modals.help_view."""

from synthea.modals.help_view import HELP_TIMEOUT_SECONDS, HelpView


class TestHelpView:
    def test_build_embed_shows_current_page(self):
        view = HelpView(["page one", "page two"], "Test Help")
        embed = view.build_embed()
        assert embed.title == "Test Help"
        assert embed.description == "page one"
        assert "Page 1/2" in embed.footer.text

    def test_button_states_on_first_page(self):
        view = HelpView(["a", "b", "c"], "Test")
        assert view.previous_button.disabled is True
        assert view.next_button.disabled is False

    def test_button_states_on_middle_page(self):
        view = HelpView(["a", "b", "c"], "Test")
        view.page = 1
        view._update_buttons()
        assert view.previous_button.disabled is False
        assert view.next_button.disabled is False
        assert view.build_embed().description == "b"

    def test_button_states_on_last_page(self):
        view = HelpView(["a", "b", "c"], "Test")
        view.page = 2
        view._update_buttons()
        assert view.previous_button.disabled is False
        assert view.next_button.disabled is True

    def test_single_page_disables_both_buttons(self):
        view = HelpView(["only"], "Test")
        assert view.previous_button.disabled is True
        assert view.next_button.disabled is True

    def test_num_pages_minimum_one(self):
        assert HelpView([], "Test").num_pages == 1

    def test_footer_mentions_self_deletion(self):
        view = HelpView(["x"], "Test", timeout=HELP_TIMEOUT_SECONDS)
        assert "5 minutes" in view.build_embed().footer.text

    def test_footer_hides_navigation_hint_on_single_page(self):
        view = HelpView(["only"], "Test")
        assert "flip through" not in view.build_embed().footer.text

    async def test_send_replies_and_registers_message(self):
        """The sent message is wired up to the view so it can be deleted on timeout."""

        sent = {}

        class FakeMessage:
            def __init__(self, reactions):
                self.reactions = reactions
                self.deleted = False

            async def reply(self, **kwargs):
                message = FakeMessage(self.reactions)
                message.kwargs = kwargs
                sent["message"] = message
                return message

            async def add_reaction(self, emoji):
                self.reactions.append(emoji)

        source = FakeMessage(reactions=[])
        await HelpView.send(source, ["one", "two"], "Test")

        message = sent["message"]
        assert message.kwargs["mention_author"] is True
        assert message.kwargs["view"].message is message
        assert "🗑️" in message.reactions
        assert message.kwargs["embed"].description == "one"
