from typing import List, Dict, Any, Optional, Union, Type, TypeVar, Generic, cast
import numpy as np
import faiss
import pickle
import os
import uuid
from datetime import datetime
import logging
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel

# Import the note types from cornell_zettel_memory_system
from cornell_zettel_memory_system import CornellMethodNote, ZettelNote
# Import SearchReranker
from utils.search_reranker import SearchReranker

logger = logging.getLogger(__name__)

# Generic type for note objects
T = TypeVar('T', bound=BaseModel)

class FaissRetriever(Generic[T]):
    """Vector database retrieval using FAISS optimized for Cornell and Zettel notes"""
    def __init__(self, 
                 note_class: Type[T],
                 model_name: str = 'nomic-embed-text',
                 index_path: Optional[str] = None,
                 use_reranker: bool = True,
                 reranker_config: Optional[Dict[str, float]] = None):
        """Initialize FAISS retriever for specific note types.
        
        Args:
            note_class: The class of notes to index (CornellMethodNote or ZettelNote)
            model_name: Name of the Ollama embedding model
            index_path: Path to load existing FAISS index (if None, uses default path)
            use_reranker: Whether to use the SearchReranker for search results
            reranker_config: Configuration for the SearchReranker weights
        """
        self.note_class = note_class
        self.class_name = note_class.__name__
        
        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(
            model=model_name
        )
        
        # Get embedding dimension by embedding a test string
        test_embedding = self.embeddings.embed_query("test")
        self.embedding_dim = len(test_embedding)
        
        # Initialize storage for notes and their IDs
        self.notes = {}  # Store actual note objects
        self.note_ids = []  # Store IDs in order they were added
        self.metadata_dict = {}  # Store additional metadata
        
        # Set default index path if None is provided
        if index_path is None:
            # Create a default path in the user's home directory
            home_dir = os.path.expanduser("~")
            default_dir = os.path.join(home_dir, ".llm_memory_agent", "faiss_indices")
            os.makedirs(default_dir, exist_ok=True)
            self.index_path = os.path.join(default_dir, f"faiss_index_{self.class_name}_{model_name}")
            logger.info(f"Using default FAISS index path: {self.index_path}")
        else:
            self.index_path = index_path
        
        # Initialize or load FAISS index
        if os.path.exists(f"{self.index_path}.index"):
            self._load_index(self.index_path)
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info(f"Created new FAISS index for {self.class_name} at {self.index_path}")
        
        # Initialize SearchReranker with default or custom configuration
        self.use_reranker = use_reranker
        if use_reranker:
            if reranker_config is None:
                # Default configuration
                self.reranker = SearchReranker(
                    recency_weight=0.2,
                    retrieval_count_weight=0.1,
                    keyword_match_weight=0.2,
                    text_overlap_weight=0.1
                )
            else:
                # Custom configuration
                self.reranker = SearchReranker(
                    recency_weight=reranker_config.get('recency_weight', 0.2),
                    retrieval_count_weight=reranker_config.get('retrieval_count_weight', 0.1),
                    keyword_match_weight=reranker_config.get('keyword_match_weight', 0.2),
                    text_overlap_weight=reranker_config.get('text_overlap_weight', 0.1),
                    custom_scoring_fn=reranker_config.get('custom_scoring_fn', None)
                )
            logger.info(f"Initialized SearchReranker for {self.class_name}")
        else:
            self.reranker = None
            logger.info(f"SearchReranker disabled for {self.class_name}")
    
    def _normalize_embedding(self, embedding: List[float]) -> np.ndarray:
        """Normalize embedding vector to unit length.
        
        Args:
            embedding: The embedding vector to normalize
            
        Returns:
            Normalized embedding as numpy array
        """
        embedding_array = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            embedding_array = embedding_array / norm
        return embedding_array
    
    def add_document(self, note: T, doc_id: str = None) -> str:
        """Add a note to the FAISS index.
        
        Args:
            note: CornellMethodNote or ZettelNote object to add
            doc_id: Unique identifier for the document (uses note.id if not provided)
            
        Returns:
            The document ID
        """
        # Use the note's ID if no doc_id is provided
        if doc_id is None:
            doc_id = note.id
        
        # Store the note object and its ID
        self.notes[doc_id] = note
        self.note_ids.append(doc_id)
        
        # Extract content for embedding
        content = note.content
        
        # Create metadata dictionary from note attributes
        metadata = note.model_dump()
        # Remove content from metadata as it's already embedded
        if 'content' in metadata:
            metadata['content_preview'] = metadata['content'][:100] + "..." if len(metadata['content']) > 100 else metadata['content']
        
        # Store metadata
        self.metadata_dict[doc_id] = metadata
        
        # Generate embedding, normalize it, and add to index
        embedding = self.embeddings.embed_query(content)
        normalized_embedding = self._normalize_embedding(embedding)
        self.index.add(np.array([normalized_embedding]))
        
        return doc_id
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the index.
        
        Note: FAISS doesn't support direct deletion. This implementation marks documents
        as deleted and rebuilds the index when necessary.
        
        Args:
            doc_id: ID of document to delete
            
        Returns:
            Whether the deletion was successful
        """
        if doc_id in self.note_ids:
            idx = self.note_ids.index(doc_id)
            # Mark as deleted by setting to None (will be filtered during rebuild)
            self.note_ids[idx] = None
            if doc_id in self.notes:
                del self.notes[doc_id]
            if doc_id in self.metadata_dict:
                del self.metadata_dict[doc_id]
            
            # Rebuild index if too many deletions (over 25%)
            if self.note_ids.count(None) > len(self.note_ids) * 0.25:
                self._rebuild_index()
            
            return True
        return False
    
    def _rebuild_index(self):
        """Rebuild the FAISS index after deletions."""
        # Filter out None values
        valid_ids = [doc_id for doc_id in self.note_ids if doc_id is not None]
        
        if not valid_ids:
            # No documents left
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.note_ids = []
            return
        
        # Create new index
        new_index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add embeddings to new index
        for doc_id in valid_ids:
            note = self.notes[doc_id]
            embedding = self.embeddings.embed_query(note.content)
            normalized_embedding = self._normalize_embedding(embedding)
            new_index.add(np.array([normalized_embedding]))
        
        # Update instance variables
        self.index = new_index
        self.note_ids = valid_ids
        
        logger.info(f"FAISS index rebuilt with {len(self.note_ids)} documents")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar notes.
        
        Args:
            query: Query text
            k: Number of results to return
        
        Returns:
            List of dicts with note objects and similarity scores
        """
        if not self.note_ids:
            return []
        
        # Get query embedding and normalize it
        query_embedding = self.embeddings.embed_query(query)
        normalized_query = self._normalize_embedding(query_embedding)
        
        # Search FAISS index
        # Retrieve more results if reranking to ensure we have enough after filtering
        search_k = min(k * 2 if self.use_reranker else k, len(self.note_ids))
        distances, indices = self.index.search(np.array([normalized_query]), search_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.note_ids) and idx >= 0:  # Valid index check
                doc_id = self.note_ids[idx]
                if doc_id is None:  # Skip deleted documents
                    continue
                
                note = self.notes[doc_id]
                
                # Update retrieval count and last accessed time
                if hasattr(note, 'retrieval_count'):
                    note.retrieval_count += 1
                if hasattr(note, 'accessed_at'):
                    note.accessed_at = datetime.now().isoformat()
                
                # Calculate similarity score (convert distance to similarity)
                similarity_score = float(1.0 / (1.0 + distances[0][i]))
                
                results.append({
                    'id': doc_id,
                    'note': note,
                    'score': similarity_score
                })
        
        # Apply reranking if enabled and we have results
        if self.use_reranker and self.reranker and results:
            results = self.reranker.rerank(query, results)
        
        return results[:k]
    
    def get_note_by_id(self, doc_id: str) -> Optional[T]:
        """Retrieve a note by its ID.
        
        Args:
            doc_id: The ID of the note to retrieve
            
        Returns:
            The note object if found, None otherwise
        """
        return self.notes.get(doc_id)
    
    def save_index(self, path: str = None):
        """Save the FAISS index and associated data to disk.
        
        Args:
            path: Directory path to save the index
        """
        if path is None:
            path = self.index_path
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.index")
        
        # Save notes, IDs and metadata
        with open(f"{path}.data", 'wb') as f:
            pickle.dump({
                'notes': self.notes,
                'note_ids': self.note_ids,
                'metadata': self.metadata_dict,
                'class_name': self.class_name
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
        
        # Load notes and metadata
        try:
            with open(f"{path}.data", 'rb') as f:
                data = pickle.load(f)
                self.notes = data['notes']
                self.note_ids = data['note_ids']
                self.metadata_dict = data['metadata']
                loaded_class_name = data.get('class_name')
                
                # Verify the loaded data matches the expected note class
                if loaded_class_name != self.class_name:
                    logger.warning(f"Loaded index contains {loaded_class_name} notes, but this retriever is configured for {self.class_name}")
            
            logger.info(f"Loaded FAISS index with {len(self.note_ids)} {self.class_name} notes")
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"Error loading FAISS data: {e}")
            # Initialize empty data structures
            self.notes = {}
            self.note_ids = []
            self.metadata_dict = {}

    def update_note(self, note: T) -> bool:
        """Update an existing note in the index.
        
        Args:
            note: The updated note object
        
        Returns:
            True if the note was updated, False otherwise
        """
        doc_id = note.id
        
        # Check if the note exists
        if doc_id not in self.notes:
            logger.warning(f"Note with ID {doc_id} not found in index")
            return False
        
        # Update the note object
        self.notes[doc_id] = note
        
        # Update metadata
        metadata = note.model_dump()
        if 'content' in metadata:
            metadata['content_preview'] = metadata['content'][:100] + "..." if len(metadata['content']) > 100 else metadata['content']
        self.metadata_dict[doc_id] = metadata
        
        # No need to update the embedding as the content hasn't changed
        # If content has changed significantly, it's better to delete and re-add the note
        
        logger.info(f"Note with ID {doc_id} updated in index")
        return True


class CornellNoteRetriever(FaissRetriever[CornellMethodNote]):
    """Specialized retriever for Cornell Method Notes"""
    def __init__(self, model_name: str = 'nomic-embed-text', index_path: Optional[str] = None):
        super().__init__(CornellMethodNote, model_name, index_path)
        
    def get_related_zettel_notes(self, cornell_note_id: str, zettel_retriever: 'ZettelNoteRetriever') -> List[ZettelNote]:
        """Get all Zettel notes linked to a Cornell note.
        
        Args:
            cornell_note_id: ID of the Cornell note
            zettel_retriever: Retriever containing Zettel notes
            
        Returns:
            List of linked Zettel notes
        """
        cornell_note = self.get_note_by_id(cornell_note_id)
        if not cornell_note:
            return []
            
        zettel_notes = []
        for zettel_id in cornell_note.zettle_ids:
            zettel_note = zettel_retriever.get_note_by_id(zettel_id)
            if zettel_note:
                zettel_notes.append(zettel_note)
                
        return zettel_notes
    
    def get_notes_by_source_id(self, source_id: str) -> List[CornellMethodNote]:
        """Get all Cornell notes that were generated from a specific source document.
        
        Args:
            source_id: ID of the source document
            
        Returns:
            List of Cornell notes generated from the source
        """
        notes_from_source = []
        for note_id, note in self.notes.items():
            if note.source_id == source_id:
                notes_from_source.append(note)
        
        return notes_from_source
    
    def update_source_id(self, note_id: str, source_id: str) -> bool:
        """Update the source_id of a Cornell note.
        
        Args:
            note_id: ID of the Cornell note to update
            source_id: New source ID to set
            
        Returns:
            Whether the update was successful
        """
        note = self.get_note_by_id(note_id)
        if note:
            note.source_id = source_id
            # Update the metadata dictionary as well
            if note_id in self.metadata_dict:
                self.metadata_dict[note_id]['source_id'] = source_id
            return True
        return False
    
    def get_source_id(self, note_id: str) -> Optional[str]:
        """Get the source_id of a Cornell note.
        
        Args:
            note_id: ID of the Cornell note
            
        Returns:
            The source_id if found, None otherwise
        """
        note = self.get_note_by_id(note_id)
        return note.source_id if note else None


class ZettelNoteRetriever(FaissRetriever[ZettelNote]):
    """Specialized retriever for Zettel Notes"""
    def __init__(self, model_name: str = 'nomic-embed-text', index_path: Optional[str] = None):
        super().__init__(ZettelNote, model_name, index_path)
        
    def get_linked_notes(self, zettel_id: str) -> List[ZettelNote]:
        """Get all Zettel notes linked to a specific Zettel note.
        
        Args:
            zettel_id: ID of the Zettel note
            
        Returns:
            List of linked Zettel notes
        """
        zettel_note = self.get_note_by_id(zettel_id)
        if not zettel_note:
            return []
            
        linked_notes = []
        for linked_id in zettel_note.links:
            linked_note = self.get_note_by_id(linked_id)
            if linked_note:
                linked_notes.append(linked_note)
                
        return linked_notes
    
    def get_parent_cornell_note(self, zettel_id: str, cornell_retriever: CornellNoteRetriever) -> Optional[CornellMethodNote]:
        """Get the parent Cornell note of a Zettel note.
        
        Args:
            zettel_id: ID of the Zettel note
            cornell_retriever: Retriever containing Cornell notes
            
        Returns:
            Parent Cornell note if found, None otherwise
        """
        zettel_note = self.get_note_by_id(zettel_id)
        if not zettel_note or not zettel_note.cornell_id:
            return None
            
        return cornell_retriever.get_note_by_id(zettel_note.cornell_id)
    
    def get_notes_by_type(self, note_type: str) -> List[ZettelNote]:
        """
        Get all notes of a specific type.
        
        Args:
            note_type: Type of notes to retrieve (e.g., "synthesis", "standard")
            
        Returns:
            List of ZettelNote objects of the specified type
        """
        result = []
        for note_id in self.note_ids:
            note = self.get_note_by_id(note_id)
            if note and note.type == note_type:
                result.append(note)
        return result