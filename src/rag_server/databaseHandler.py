import chromadb
from importlib_metadata import metadata
from sentence_transformers import SentenceTransformer
import os


class DatabaseHandler:
    
    """Handles storage and retrieval of documents from a database."""
    
    def __init__(self, path, model_name, logger):
        
        """Initializes the database handler with a directory and embedding model name.

        Args:
            path (str): Path to the database persistence directory.
            model_name (str): Name of the embedding model to be used.
        """
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name="ias_documents_store")
        self.model = SentenceTransformer(model_name)
        self.logger = logger
        # E5 family models expect "passage: " / "query: " prefixes for best retrieval.
        # For non-E5 models, prefixes are left empty so behavior is identical to before.
        self._is_e5 = "e5" in (model_name or "").lower()
        self._passage_prefix = "passage: " if self._is_e5 else ""
        self._query_prefix = "query: " if self._is_e5 else ""
        if self._is_e5:
            self.logger.info(
                f"E5 model detected ({model_name}); using 'query:'/'passage:' prefixes."
            )
        self.logger.info("DatabaseHandler initialized successfully.")

    def store_documents(self, chunks, metadatas):

        """Stores processed documents into the database.

        Args:
            chunks (List[str]): A list of documents' chunks to store.
            metadatas (List[Dict]): A list of metadata dictionaries for each chunk.
        """
        
        for idx, chunk in enumerate(chunks):
            text_to_embed = f"{self._passage_prefix}{chunk}"
            embedding = self.model.encode(text_to_embed, normalize_embeddings=True).tolist() # using normalize for better results
            self.collection.add(
                ids=[f"doc_{idx}"],
                documents=[chunk],
                metadatas=metadatas[idx],
                embeddings=[embedding]
            )
        self.logger.info(f"Stored {len(chunks)} documents in the database.")

    def query(self, query_text, top_k=5):
        
        """Queries the database to retrieve relevant documents.

        Args:
            query_text (str): The query or question to match.
            top_k (int): Number of top documents to retrieve.

        Returns:
            List[str]: List of relevant document texts.
        """
        self.logger.info(f"Executing Query Retrieval.")
        text_to_embed = f"{self._query_prefix}{query_text}"
        query_embedding = self.model.encode([text_to_embed], normalize_embeddings=True).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)
        self.logger.info(f"Query Retrieval successful.")
        return results["documents"]



def get_embedding_dim():
    from chromadb import PersistentClient

    # Load your DB
    client = PersistentClient(path="./src/rag_server/chroma_db")

    collection = client.get_collection("ias_documents_store")
    print("Collection name:", collection.name)
    print("Number of items in collection:", collection.count())
    print(client.list_collections())


    # Peek one item to get the embedding
    sample = collection.peek(1)

    # Print embedding vector length
    embedding_dim = len(sample['embeddings'][0])
    print("Embedding dimension:", embedding_dim)
    
      
if __name__ == "__main__":
    get_embedding_dim()
