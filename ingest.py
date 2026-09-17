import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import embeddings


def ingest_document(file_path: str, persist_directory: str = "chroma_db"):
    """
    Loads the PDF at `file_path`, splits it into chunks, and writes the
    embeddings into a Chroma vectorstore at `persist_directory`.
    """
    # Chroma caches its internal DB connection per-path for the life of the
    # process. Since Streamlit reruns this script in the same process,
    # deleting/recreating `persist_directory` between uploads leaves that
    # cache pointing at stale, deleted files -> "readonly database" errors.
    # Clearing it here forces a fresh connection every time we ingest.
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    print("Loading PDF...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    print("Splitting document...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,   # 500 characters per chunk
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Creating vector database...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("Done!")


if __name__ == "__main__":
    ingest_document("data/demo.pdf")