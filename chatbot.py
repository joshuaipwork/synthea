"""
The discord client which contains the bulk of the logic for the chatbot.
"""
import argparse
from typing import Any
import discord
import yaml

from model import ChattyModel

COMMAND_START_STR: str = '!syn'

# This example requires the 'message_content' intent.
class LLMClient(discord.Client):
    model = None
    async def setup_hook(self):
        """
        When the bot is started and logs in, load the model.
        """
        self.model = ChattyModel()
        self.model.load_model()

    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message: discord.Message):
        if message.author == self.user:
            return

        if message.webhook_id:
            return

        if message.content.startswith(COMMAND_START_STR):
            # let the user know that we are working on their command
            await message.add_reaction("⏳")

            try:
                command = message.content[len(COMMAND_START_STR) + 1:]
                split_command = command.split()

                # check if we are generating from a previously saved prompt, or a basic prompt
                if split_command[0] == '-c' or split_command[0] == '-char' or split_command[0] == '--character':
                    # expected format: [COMMAND_START_STR] [-c CHARACTER] [PROMPT...]
                    if len(split_command) < 2:
                        raise ValueError("You must specify a character to use with the prompt if you specify -c or --character")

                    character = split_command[1]
                    prompt = " ".join(split_command[2:])
                    print(f'Generating with char {character} with prompt {prompt} for from {message.author}')
                    output = self.model.generate_from_character(prompt=prompt, character=character)
                    print(output)

                    with open(f'characters/{character}.yaml', "r") as f:
                        loaded_config = yaml.safe_load(f)

                        # create a temporary webhook for the bot to speak as its character
                        if 'avatar' in loaded_config:
                            with open(f"avatars/{loaded_config['avatar']}", "rb") as avatar_file:
                                avatar = avatar_file.read()
                        else:
                            avatar = None

                        webhook: discord.Webhook = await message.channel.create_webhook(
                            name=loaded_config['name'],
                            avatar=avatar,
                            reason="Chatbot character"
                        )

                        # send the message via its character
                        await webhook.send(output)

                        # remove the webhook once done
                        await webhook.delete()

                else:
                    # expected format: [COMMAND_START_STR] [PROMPT...]
                    prompt = command
                    print(f'Generating with prompt {prompt} for from {message.author}')
                    output = self.model.generate_from_prompt(prompt)
                    print(output)
                    await message.reply(output, mention_author=True)

                # let the user know that we successfully completed their task.
                await message.add_reaction("✅")
            except Exception as err:
                # let the user know that something went wrong
                await message.add_reaction("❌")
                await message.reply(str(err), mention_author=True)
            finally:
                # let the user know that we are done with their command
                await message.remove_reaction("⏳", self.user)
            
