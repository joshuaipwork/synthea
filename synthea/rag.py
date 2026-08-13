import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,  # bulk-load a folder
    PyPDFLoader,
    TextLoader,
)
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from synthea.config import Config
from synthea.utilities import inference_logger

VALID_EXTENSIONS = [".txt", ".pdf", ".docx"]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # larger chunks preserve more context
    chunk_overlap=200,  # ~20% overlap is a good rule of thumb
    add_start_index=True,
)


def get_vectorstore(guild_id: int, user_id: int) -> Chroma:
    embeddings = OpenAIEmbeddings(
        base_url=Config().embeddings_base_url, model=Config().embeddings_model,
    )

    collection_name: str = f"rag_docs_{guild_id}"
    persist_directory: str = "./chroma_db/rag"
    if not guild_id:
        collection_name = f"rag_docs_user_{user_id}"
        persist_directory = "./chroma_db/rag/users/"

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


async def ingest_document(file_path: str, guild_id: int, user_id: int) -> None:
    """Ingests a document at the specified path into the vector store
    """
    # pick loader by extension
    ext = file_path.rsplit(".", 1)[-1].lower()
    loaders = {"pdf": PyPDFLoader, "txt": TextLoader, "docx": Docx2txtLoader}
    loader = loaders[ext](file_path)

    docs = loader.load()

    chunks = splitter.split_documents(docs)

    vectorstore: Chroma = get_vectorstore(guild_id, user_id)

    saved_chunks = await vectorstore.aadd_documents(chunks)

    inference_logger.info(f"Ingested {len(saved_chunks)} chunks from {file_path}")


WhereDocument = dict  # ChromaDB where_document filter


def build_document_filter(
    require: str | list[str] | None = None,
    exclude: str | list[str] | None = None,
    regex: str | None = None,
) -> WhereDocument | None:
    """Builds a ChromaDB where_document filter from the provided constraints.

    Args:
        require: A term or list of terms that MUST appear in the chunk (OR logic).
        exclude: A term or list of terms that MUST NOT appear in the chunk (AND logic).
        regex:   A regex pattern that the chunk must match.

    """
    clauses = []

    if require:
        terms = [require] if isinstance(require, str) else require
        if len(terms) == 1:
            clauses.append({"$contains": terms[0]})
        else:
            clauses.append({"$or": [{"$contains": t} for t in terms]})

    if exclude:
        terms = [exclude] if isinstance(exclude, str) else exclude
        for term in terms:
            clauses.append({"$not_contains": term})

    if regex:
        clauses.append({"$regex": regex})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


async def retrieve_documents(
    query: str,
    guild_id: int,
    user_id: int,
    require: str | list[str] | None = None,
    exclude: str | list[str] | None = None,
    regex: str | None = None,
) -> str:
    vectorstore: Chroma = get_vectorstore(guild_id, user_id)
    collection = vectorstore._collection
    embeddings = vectorstore.embeddings
    query_embedding = embeddings.embed_query(query)

    document_filter = build_document_filter(require, exclude, regex)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where_document=document_filter,
    )

    if not results["documents"] or not results["documents"][0]:
        return "No relevant documents found for this query."

    return "\n\n".join(
        f"Source: {meta.get('source', 'unknown')}\n-----------------------------\n{doc}"
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    )


async def delete_document(file_path: str, guild_id: int, user_id: int):
    """Deletes all the chunks in the document store that have been sourced from
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
