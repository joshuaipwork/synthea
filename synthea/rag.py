import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    WebBaseLoader,       # scrape a URL
    DirectoryLoader,     # bulk-load a folder
)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from synthea.config import Config
from synthea.utilities import inference_logger

VALID_EXTENSIONS = [".txt", ".pdf", ".docx"]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,       # characters, not tokens
    chunk_overlap=50,     # overlap to avoid cutting context at boundaries
    add_start_index=True  # adds char offset to metadata — useful for debugging
)

def get_vectorstore(guild_id: int, user_id: int) -> Chroma:
    embeddings = OpenAIEmbeddings(
        base_url=Config().embeddings_base_url,
        model=Config().embeddings_model)


    collection_name: str = f"rag_docs_{guild_id}"
    persist_directory: str = "./chroma_db/rag"
    if not guild_id:
        collection_name = f"rag_docs_user_{user_id}"
        persist_directory = "./chroma_db/rag/users/"
    
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

async def ingest_document(file_path: str, guild_id: int, user_id: int) -> None:
    """
    Ingests a document at the specified path into the vector store
    """
    # pick loader by extension
    ext = file_path.rsplit(".", 1)[-1].lower()
    loaders = {"pdf": PyPDFLoader, "txt": TextLoader, "docx": Docx2txtLoader}
    loader = loaders[ext](file_path)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # characters per chunk
        chunk_overlap=200,    # overlap to preserve context across chunk boundaries
    )
    chunks = splitter.split_documents(docs)

    vectorstore: Chroma = get_vectorstore(guild_id, user_id)

    saved_chunks = await vectorstore.aadd_documents(chunks)

    inference_logger.info(f"Ingested {len(saved_chunks)} chunks from {file_path}")

async def retrieve_documents(query: str, guild_id: int, user_id: int) -> str:
    vectorstore: Chroma = get_vectorstore(guild_id, user_id)
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 4, "score_threshold": 0.6},
    )
    docs = await retriever.ainvoke(query)

    if not docs:
        return "No relevant documents found for this query."

    return "\n\n".join(
        f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}"
        for d in docs
    )

async def delete_document(file_path: str, guild_id: int, user_id: int):
    """
    deletes all the chunks in the document store that have been sourced from
    the specified document.
    """
    collection = get_vectorstore(guild_id, user_id)._collection
    
    # find all chunks where source matches
    results = collection.get(where={"source": file_path})
    
    if not results["ids"]:
        inference_logger.warning(f"No chunks found for {file_path}")
        return
    
    collection.delete(ids=results["ids"])
    inference_logger.info(f"Deleted {len(results['ids'])} chunks for {file_path}")


def get_document_path(guild_id: int, user_id: int) -> str:
    if guild_id:
        save_directory = f"./saved_documents/{guild_id}/"
    else:
        save_directory = f"./saved_documents/users/{user_id}"

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    return save_directory
