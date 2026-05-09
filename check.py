import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# First, see what collections actually exist
print(client.list_collections())

# Then peek at the one mem0 created
collection = client.get_collection("chatbot_memories")
print(collection.count())
print(collection.peek())