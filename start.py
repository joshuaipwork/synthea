import discord
import argparse
import json
from chatbot import LLMClient

with open('secrets.json', "r") as f:
    secrets = json.load(f)

    # set up the discord bot
    intents = discord.Intents.all()
    intents.message_content = True
    intents.presences = True
    intents.members = True

    client = LLMClient(intents=intents)
    client.run(secrets['client_token'])
