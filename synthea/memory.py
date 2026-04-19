# from mem0 import Memory

# config = {
#     "vector_store": {
#         "provider": "chroma",
#         "config": {
#             "collection_name": "agent_memories",
#             "path": "./chroma_db",        # local persistent path
#             # Or for a remote Chroma server:
#             # "host": "localhost",
#             # "port": 8000,
#         }
#     },
#     "llm": {
#         "provider": "openai",             # mem0 uses openai-compat provider
#         "config": {
#             "model": "local-model",       # can be any string; llama-server ignores it
#             "openai_base_url": "http://localhost:8080/v1",
#             "api_key": "not-needed",      # required by client but unused
#             "temperature": 0.1,
#             "max_tokens": 2000,
#         }
#     },
#     "embedder": {
#         "provider": "openai",             # llama-server also serves /v1/embeddings
#         "config": {
#             "model": "local-model",
#             "openai_base_url": "http://localhost:8080/v1",
#             "api_key": "not-needed",
#         }
#     }
# }

# memory = Memory.from_config(config)