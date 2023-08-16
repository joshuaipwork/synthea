from CommandParser import ChatbotParser, CommandParser

parser = ChatbotParser()

string = "!syn help me "

try:
    args = parser.parse(string[5:])
    print(args)
except Exception as e:
    print(f'Caught exception: {e}')
print("successfully ended")