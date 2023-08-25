import discord
import yaml
from synthea.SyntheaClient import SyntheaClient

with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# set up the discord bot
intents = discord.Intents.all()
intents.message_content = True
intents.presences = True
intents.members = True

client = SyntheaClient(intents=intents)
client.run(config["client_token"])
