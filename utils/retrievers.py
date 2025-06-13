from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings
import pickle
from nltk.tokenize import word_tokenize
import os
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import faiss
import uuid
import logging

logger = logging.getLogger(__name__)

def simple_tokenize(text):
    return word_tokenize(text)

class SimpleEmbeddingRetriever:
    """Simple retriever using sentence embeddings"""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None

    def add_document(self, document: str):
        """Add a document to the retriever.

        Args:
            document: Text content to add
        """
        self.documents.append(document)
        # Update embeddings
        if len(self.documents) == 1:
            self.embeddings = self.model.encode([document])
        else:
            new_embedding = self.model.encode([document])
            self.embeddings = np.vstack([self.embeddings, new_embedding])

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of dictionaries containing document content and similarity score
        """
        if not self.documents:
            return []

        # Get query embedding
        query_embedding = self.model.encode([query])

        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'content': self.documents[idx],
                'score': float(similarities[idx])
            })

        return results

class ChromaRetriever:
    """Vector database retrieval using ChromaDB"""
    def __init__(self, collection_name: str = "memories"):
        """Initialize ChromaDB retriever.

        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.client = chromadb.Client(Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_document(self, document: str, metadata: Dict, doc_id: str):
        """Add a document to ChromaDB.

        Args:
            document: Text content to add
            metadata: Dictionary of metadata
            doc_id: Unique identifier for the document
        """
        # Convert lists to strings in metadata to comply with ChromaDB requirements
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                processed_metadata[key] = ", ".join(value)
            else:
                processed_metadata[key] = value

        self.collection.add(
            documents=[document],
            metadatas=[processed_metadata],
            ids=[doc_id]
        )

    def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB.

        Args:
            doc_id: ID of document to delete
        """
        self.collection.delete(ids=[doc_id])

    def search(self, query: str, k: int = 5):
        """Search for similar documents.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of dicts with document text and metadata
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )

        # Convert string metadata back to lists where appropriate
        if 'metadatas' in results and results['metadatas']:
            for metadata in results['metadatas']:
                for key in ['keywords', 'tags']:
                    if key in metadata and isinstance(metadata[key], str):
                        metadata[key] = [item.strip() for item in metadata[key].split(',')]

        return results


class MilvusRetriever:
    """Vector database retrieval using Milvus"""
    def __init__(self, collection_name: str = "memories", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.collection_name = collection_name
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = Milvus(
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            connection_args={"uri": "./milvus_example.db"},
            index_params={"metric_type": "L2", "index_type": "FLAT", "params": {"nlist": 1024}},
        )

    def add_document(self, document: str, metadata: Dict, doc_id: str):
        doc = Document(page_content=document, metadata=metadata)
        self.vector_store.add_documents([doc], ids=[doc_id])

    def delete_document(self, doc_id: str):
        self.vector_store.delete([doc_id])

    def search(self, query: str, k: int = 5):
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                "id": result[0].metadata.get("id", ""),
                "content": result[0].page_content,
                "metadata": result[0].metadata,
                "score": result[1],
            }
            for result in results
        ]

    def get_collection_info(self):
        return self.vector_store.col.schema


class FaissRetriever:
    """Vector database retrieval using FAISS"""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', index_path: Optional[str] = None):
        """Initialize FAISS retriever.
        
        Args:
            model_name: Name of the sentence transformer model
            index_path: Path to load existing FAISS index (if None, uses default path)
        """
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.document_ids = []
        self.metadata_dict = {}
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Set default index path if None is provided
        if index_path is None:
            # Create a default path in the user's home directory
            home_dir = os.path.expanduser("~")
            default_dir = os.path.join(home_dir, ".llm_memory_agent", "faiss_indices")
            os.makedirs(default_dir, exist_ok=True)
            self.index_path = os.path.join(default_dir, f"faiss_index_{model_name.replace('/', '_')}")
            logger.info(f"Using default FAISS index path: {self.index_path}")
        else:
            self.index_path = index_path
        
        # Initialize or load FAISS index
        if os.path.exists(f"{self.index_path}.index"):
            self._load_index(self.index_path)
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info(f"Created new FAISS index at {self.index_path}")
    
    def add_document(self, document: str, metadata: Dict = None, doc_id: str = None):
        """Add a document to the FAISS index.
        
        Args:
            document: Text content to add
            metadata: Dictionary of metadata
            doc_id: Unique identifier for the document (generated if not provided)
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        # Store document and metadata
        self.documents.append(document)
        self.document_ids.append(doc_id)
        
        if metadata:
            self.metadata_dict[doc_id] = metadata
        
        # Generate embedding and add to index
        embedding = self.model.encode([document])
        self.index.add(np.array(embedding, dtype=np.float32))
        
        return doc_id
    
    def delete_document(self, doc_id: str):
        """Delete a document from the index.
        
        Note: FAISS doesn't support direct deletion. This implementation marks documents
        as deleted and rebuilds the index when necessary.
        
        Args:
            doc_id: ID of document to delete
        """
        if doc_id in self.document_ids:
            idx = self.document_ids.index(doc_id)
            # Mark as deleted by setting to None (will be filtered during rebuild)
            self.documents[idx] = None
            self.document_ids[idx] = None
            if doc_id in self.metadata_dict:
                del self.metadata_dict[doc_id]
            
            # Rebuild index if too many deletions (over 25%)
            if self.documents.count(None) > len(self.documents) * 0.25:
                self._rebuild_index()
            
            return True
        return False
    
    def _rebuild_index(self):
        """Rebuild the FAISS index after deletions."""
        # Filter out None values
        valid_docs = [(i, doc, doc_id) for i, (doc, doc_id) in 
                     enumerate(zip(self.documents, self.document_ids)) 
                     if doc is not None]
        
        if not valid_docs:
            # No documents left
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.documents = []
            self.document_ids = []
            return
        
        # Unpack the filtered data
        indices, filtered_docs, filtered_ids = zip(*valid_docs)
        
        # Create new index
        new_index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add embeddings to new index
        embeddings = self.model.encode(filtered_docs)
        new_index.add(np.array(embeddings, dtype=np.float32))
        
        # Update instance variables
        self.index = new_index
        self.documents = list(filtered_docs)
        self.document_ids = list(filtered_ids)
        
        logger.info(f"FAISS index rebuilt with {len(self.documents)} documents")
    
    def search(self, query: str, k: int = 5):
        """Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of dicts with document content, metadata, and similarity score
        """
        if not self.documents:
            return []
        
        # Get query embedding
        query_embedding = self.model.encode([query])
        
        # Search FAISS index
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:  # Valid index check
                doc_id = self.document_ids[idx]
                metadata = self.metadata_dict.get(doc_id, {})
                
                results.append({
                    'id': doc_id,
                    'content': self.documents[idx],
                    'metadata': metadata,
                    'score': float(1.0 / (1.0 + distances[0][i]))  # Convert distance to similarity score
                })
        
        return results
    
    def save_index(self, path: str = None):
        """Save the FAISS index and associated data to disk.
        
        Args:
            path: Directory path to save the index
        """
        if path is None:
            path = self.index_path or "faiss_index"
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.index")
        
        # Save documents, IDs and metadata
        with open(f"{path}.data", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'document_ids': self.document_ids,
                'metadata': self.metadata_dict
            }, f)
        
        logger.info(f"FAISS index saved to {path}")
    
    def _load_index(self, path: str):
        """Load FAISS index and associated data from disk.
        
        Args:
            path: Path to the saved index
        """
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.index")
        self.index_path = path
        
        # Load documents and metadata
        try:
            with open(f"{path}.data", 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.document_ids = data['document_ids']
                self.metadata_dict = data['metadata']
            
            logger.info(f"Loaded FAISS index with {len(self.documents)} documents")
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"Error loading FAISS data: {e}")
            # Initialize empty data structures
            self.documents = []
            self.document_ids = []
            self.metadata_dict = {}