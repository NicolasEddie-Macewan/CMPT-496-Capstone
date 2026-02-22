"""
@file: vector_store_wrapper.py
@description: A wrapper around ChromaDB and HuggingFaceEmbeddings to provide a clean interface for LangChain/LangGraph
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import os

class ChromaManager:
    """
    @brief A wrapper class to manage interactions with a ChromaDB vector store using HuggingFace embeddings.
    @param db_path The file path to the persistent ChromaDB directory
    @param collection_name The name of the ChromaDB collection to connect to
    """
    def __init__(self, db_path: str, collection_name: str):
        # 1. Initialize the Embedding Model (Downloads on first run)
        # This model must match what was used to create the DB originally.
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 2. Connect to the existing persistent Chroma Client
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"No ChromaDB found at {db_path}")
            
        self.client = chromadb.PersistentClient(path=db_path)

        # 3. Initialize the LangChain VectorStore wrapper
        self.vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )

    def get_retriever(self, k=3):
        """
        @brief Provides a retriever interface for LangChain that abstracts away the underlying vector store.
        @param k The number of top results to return for each query
        """
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    # AI generated search method - Leaving it here for now
    # TODO: Verify if this is sufficient
    def similarity_search(self, query: str, k=3):
        """
        @brief A direct method to perform a similarity search on the vector store without using the retriever interface.
        @param query The input query string to search for
        @param k The number of top results to return
        """
        return self.vector_store.similarity_search(query, k=k)
