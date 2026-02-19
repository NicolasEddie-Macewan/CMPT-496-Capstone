import chromadb
from pathlib import Path
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

def test_top_k():
    embedding = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # Absolute path verification
    db_path = (Path(__file__).parent.parent.parent / "vectorStores").resolve()

    client = chromadb.PersistentClient(path=str(db_path))

    # Target Humanizer_db collection for testing
    target_name = "Humanizer_db"

    collection = client.get_collection(name=target_name, embedding_function=embedding)
    
    # Crete random query to test top-k retrieval
    query = "How does Humanizer round times?"
    results = collection.query(query_texts=[query], n_results=5)

    print(f"\nQuery: {query}")
    for i, doc in enumerate(results['documents'][0]):
        print(f"\nResult {i+1}: {doc}")

if __name__ == "__main__":
    test_top_k()