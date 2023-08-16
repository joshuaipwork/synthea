# What is a character?
A 'character' is just a set of settings that are used with the model when generating text.

These settings include system prompts, which give context to how the bot should behave when talking.
System prompts are always included in the context. Characters also include generation settings such as
temperature, top_p, top_k, and maximum new tokens. These generation settings will be applied
when the bot speaks as a character. Finally, characters may include an avatar and a display name, which are
shown by the bot when speaking as that character.

Characters have names. The name of the character is the name of yaml file in the characters/ folder which contains the details about the character. No two characters may have the same name in the same server.

Characters are guild-specific. The bot may be hosted on multiple guilds at once,
and some characters may not be appropriate for all guilds. In DMs, you can invoke any character that you've made. 

Characters are also author-specific, and may only be updated by the user who
originally created that character.