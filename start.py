import discord
import argparse
import json
from SyntheaClient import SyntheaClient

with open('secrets.json', "r") as f:
    secrets = json.load(f)

    # set up the discord bot
    intents = discord.Intents.all()
    intents.message_content = True
    intents.presences = True
    intents.members = True

    client = SyntheaClient(intents=intents)
    client.run(secrets['client_token'])
