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
        self.logger.info("DatabaseHandler initialized successfully.")

    def store_documents(self, chunks, metadatas):

        """Stores processed documents into the database.

        Args:
            chunks (List[str]): A list of documents' chunks to store.
            metadatas (List[Dict]): A list of metadata dictionaries for each chunk.
        """
        
        for idx, chunk in enumerate(chunks):
            embedding = self.model.encode(chunk).tolist()
            self.collection.add(
                ids=[f"doc_{idx}"],
                documents=[chunk],
                metadatas=metadatas[idx],
                embeddings=[embedding]
            )
        self.logger.info(f"Stored {len(chunks)} documents in the database.")

    def query(self, query_text, top_k=3):
        
        """Queries the database to retrieve relevant documents.

        Args:
            query_text (str): The query or question to match.
            top_k (int): Number of top documents to retrieve.

        Returns:
            List[str]: List of relevant document texts.
        """
        self.logger.info(f"Executing query: {query_text}")
        query_embedding = self.model.encode([query_text]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)
        self.logger.info(f"Query Retrieval successful.")
        return results["documents"]
