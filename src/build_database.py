import chromadb
import sys
from pathlib import Path
from utils.tree_parse import *
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

def build_database(source_path: str) -> None:
    """
    @brief Assembles a local ChromaDB database from a directory of source files.
    """
    script_dir = Path(__file__).parent
    source_dir = (Path(script_dir) / source_path).resolve()
    db_dir = (Path(script_dir) / "./vectorStores").resolve()
    db_name = f"{source_dir.name}_db"

    # Create the vectoreStores directory if it doesn't exist
    if not db_dir.exists():
        db_dir.mkdir(parents=True, exist_ok=True)

    # Initialize embedding function
    embedding = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=str(db_dir))

    # Create or get the collection
    collection = client.get_or_create_collection(name=db_name, embedding_function = embedding)

    # Parse and chunk the source files
    bundles = parse_dir(source_dir)
    for bundle in bundles:
        chunks = get_chunks(bundle)
        for chunk in chunks:
            # Create unique ID for the chunk
            unique_id = f"{chunk['file']}_{chunk['class']}_{chunk['name']}"

            # Create string for embedder
            embedded_string = f"Namespace: {chunk['namespace']}\nClass: {chunk['class']}\nType: {chunk['type']}\nCode: {chunk['code']}"

            # Prepare metadata
            
            collection.upsert(

            )


if __name__ == "__main__":
    pass