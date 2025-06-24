from typing import List, Dict, Any, Optional, Union, Type, TypeVar, Generic, cast, Tuple
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
from utils.cornell_zettel_memory_system import CornellMethodNote, ZettelNote
# Import SearchReranker
from utils.search_reranker import SearchReranker

# Set up logger for this module with a more descriptive name
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
        """Initialize FAISS retriever for specific note types."""
        self.note_class = note_class
        self.class_name = note_class.__name__
        
        logger.info(f"Initializing {self.class_name} retriever with model '{model_name}'")
        
        # Initialize Ollama embeddings
        try:
            self.embeddings = OllamaEmbeddings(model=model_name)
            logger.debug(f"Successfully initialized OllamaEmbeddings with model '{model_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize OllamaEmbeddings: {str(e)}")
            raise
        
        # Get embedding dimension by embedding a test string
        try:
            test_embedding = self.embeddings.embed_query("test")
            self.embedding_dim = len(test_embedding)
            logger.debug(f"Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to determine embedding dimension: {str(e)}")
            raise
        
        # Initialize storage for notes and their IDs
        self.notes = {}  # Store actual note objects
        self.note_ids = []  # Store IDs in order they were added
        self.metadata_dict = {}  # Store additional metadata
        logger.debug("Initialized empty storage containers for notes and metadata")
        
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
            logger.info(f"Using provided FAISS index path: {self.index_path}")
        
        # Initialize or load FAISS index
        if os.path.exists(f"{self.index_path}.index"):
            logger.info(f"Found existing FAISS index at {self.index_path}.index, loading...")
            self._load_index(self.index_path)
        else:
            logger.info(f"No existing index found, creating new FAISS index for {self.class_name}")
            try:
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                logger.info(f"Created new FAISS index for {self.class_name} at {self.index_path}")
            except Exception as e:
                logger.error(f"Failed to create FAISS index: {str(e)}")
                raise
        
        # Initialize SearchReranker with default or custom configuration
        self.use_reranker = use_reranker
        if use_reranker:
            logger.info("Initializing SearchReranker for result reranking")
            try:
                if reranker_config is None:
                    # Default configuration
                    logger.debug("Using default reranker configuration")
                    self.reranker = SearchReranker(
                        recency_weight=0.2,
                        retrieval_count_weight=0.1,
                        keyword_match_weight=0.2,
                        text_overlap_weight=0.1
                    )
                else:
                    # Custom configuration
                    logger.debug(f"Using custom reranker configuration: {reranker_config}")
                    self.reranker = SearchReranker(
                        recency_weight=reranker_config.get('recency_weight', 0.2),
                        retrieval_count_weight=reranker_config.get('retrieval_count_weight', 0.1),
                        keyword_match_weight=reranker_config.get('keyword_match_weight', 0.2),
                        text_overlap_weight=reranker_config.get('text_overlap_weight', 0.1),
                        custom_scoring_fn=reranker_config.get('custom_scoring_fn', None)
                    )
                logger.info(f"Successfully initialized SearchReranker for {self.class_name}")
            except Exception as e:
                logger.error(f"Failed to initialize SearchReranker: {str(e)}")
                self.reranker = None
                self.use_reranker = False
        else:
            self.reranker = None
            logger.info(f"SearchReranker disabled for {self.class_name}")
        
        logger.info(f"{self.class_name} retriever initialization complete")
    
    def _normalize_embedding(self, embedding: List[float]) -> np.ndarray:
        """Normalize embedding vector to unit length."""
        try:
            embedding_array = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                embedding_array = embedding_array / norm
            return embedding_array
        except Exception as e:
            logger.error(f"Error normalizing embedding: {str(e)}")
            raise
    
    def add_document(self, note: T, doc_id: str = None) -> str:
        """Add a note to the FAISS index."""
        # Use the note's ID if no doc_id is provided
        if doc_id is None:
            doc_id = note.id
        
        logger.info(f"Adding {self.class_name} document with ID: {doc_id}")
        
        try:
            # Store the note object and its ID
            self.notes[doc_id] = note
            self.note_ids.append(doc_id)
            
            # Extract content for embedding
            content = note.content
            content_preview = content[:50] + "..." if len(content) > 50 else content
            logger.debug(f"Document content preview: {content_preview}")
            
            # Create metadata dictionary from note attributes
            metadata = note.model_dump()
            # Remove content from metadata as it's already embedded
            if 'content' in metadata:
                metadata['content_preview'] = metadata['content'][:100] + "..." if len(metadata['content']) > 100 else metadata['content']
            
            # Store metadata
            self.metadata_dict[doc_id] = metadata
            logger.debug(f"Stored metadata for document {doc_id}")
            
            # Generate embedding, normalize it, and add to index
            logger.debug(f"Generating embedding for document {doc_id}")
            embedding = self.embeddings.embed_query(content)
            normalized_embedding = self._normalize_embedding(embedding)
            self.index.add(np.array([normalized_embedding]))
            
            logger.info(f"Successfully added document {doc_id} to index")
            return doc_id
        except Exception as e:
            logger.error(f"Error adding document {doc_id} to index: {str(e)}")
            # Clean up if there was an error
            if doc_id in self.notes:
                del self.notes[doc_id]
            if doc_id in self.note_ids:
                self.note_ids.remove(doc_id)
            if doc_id in self.metadata_dict:
                del self.metadata_dict[doc_id]
            raise
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the index."""
        logger.info(f"Attempting to delete document with ID: {doc_id}")
        
        if doc_id in self.note_ids:
            try:
                idx = self.note_ids.index(doc_id)
                # Mark as deleted by setting to None (will be filtered during rebuild)
                self.note_ids[idx] = None
                logger.debug(f"Marked document {doc_id} as deleted at index {idx}")
                
                if doc_id in self.notes:
                    del self.notes[doc_id]
                if doc_id in self.metadata_dict:
                    del self.metadata_dict[doc_id]
                
                # Rebuild index if too many deletions (over 25%)
                deletion_count = self.note_ids.count(None)
                total_count = len(self.note_ids)
                deletion_percentage = (deletion_count / total_count) * 100 if total_count > 0 else 0
                
                logger.debug(f"Current deletion stats: {deletion_count}/{total_count} documents deleted ({deletion_percentage:.2f}%)")
                
                if deletion_count > total_count * 0.25:
                    logger.info(f"Deletion threshold reached ({deletion_percentage:.2f}% > 25%), rebuilding index")
                    self._rebuild_index()
                
                logger.info(f"Successfully deleted document {doc_id}")
                return True
            except Exception as e:
                logger.error(f"Error deleting document {doc_id}: {str(e)}")
                return False
        else:
            logger.warning(f"Document {doc_id} not found in index, nothing to delete")
            return False
    
    def _rebuild_index(self):
        """Rebuild the FAISS index after deletions."""
        logger.info("Rebuilding FAISS index")
        
        try:
            # Filter out None values
            valid_ids = [doc_id for doc_id in self.note_ids if doc_id is not None]
            logger.debug(f"Found {len(valid_ids)} valid documents for index rebuild")
            
            if not valid_ids:
                # No documents left
                logger.info("No valid documents remaining, creating empty index")
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.note_ids = []
                return
            
            # Create new index
            logger.debug(f"Creating new FAISS index with dimension {self.embedding_dim}")
            new_index = faiss.IndexFlatL2(self.embedding_dim)
            
            # Add embeddings to new index
            for i, doc_id in enumerate(valid_ids):
                logger.debug(f"Re-adding document {doc_id} ({i+1}/{len(valid_ids)})")
                note = self.notes[doc_id]
                embedding = self.embeddings.embed_query(note.content)
                normalized_embedding = self._normalize_embedding(embedding)
                new_index.add(np.array([normalized_embedding]))
            
            # Update instance variables
            self.index = new_index
            self.note_ids = valid_ids
            
            logger.info(f"FAISS index successfully rebuilt with {len(self.note_ids)} documents")
        except Exception as e:
            logger.error(f"Error rebuilding FAISS index: {str(e)}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar notes."""
        logger.info(f"Searching for query: '{query[:50]}...' with k={k}")
        
        if not self.note_ids:
            logger.warning("Index is empty, returning empty results")
            return []
        
        try:
            # Get query embedding and normalize it
            logger.debug("Generating embedding for query")
            query_embedding = self.embeddings.embed_query(query)
            normalized_query = self._normalize_embedding(query_embedding)
            
            # Search FAISS index
            # Retrieve more results if reranking to ensure we have enough after filtering
            search_k = min(k * 2 if self.use_reranker else k, len(self.note_ids))
            logger.debug(f"Searching index with k={search_k} (internal)")
            
            distances, indices = self.index.search(np.array([normalized_query]), search_k)
            logger.debug(f"FAISS search returned {len(indices[0])} results")
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.note_ids) and idx >= 0:  # Valid index check
                    doc_id = self.note_ids[idx]
                    if doc_id is None:  # Skip deleted documents
                        logger.debug(f"Skipping deleted document at index {idx}")
                        continue
                    
                    note = self.notes[doc_id]
                    
                    # Update retrieval count and last accessed time
                    if hasattr(note, 'retrieval_count'):
                        note.retrieval_count += 1
                        logger.debug(f"Incremented retrieval count for {doc_id} to {note.retrieval_count}")
                    if hasattr(note, 'accessed_at'):
                        note.accessed_at = datetime.now().isoformat()
                        logger.debug(f"Updated accessed_at for {doc_id} to {note.accessed_at}")
                    
                    # Calculate similarity score (convert distance to similarity)
                    similarity_score = float(1.0 / (1.0 + distances[0][i]))
                    
                    results.append({
                        'id': doc_id,
                        'note': note,
                        'score': similarity_score
                    })
                    logger.debug(f"Added result {doc_id} with score {similarity_score:.4f}")
            
            logger.debug(f"Collected {len(results)} valid results before reranking")
            
            # Apply reranking if enabled and we have results
            if self.use_reranker and self.reranker and results:
                logger.debug("Applying reranking to search results")
                results = self.reranker.rerank(query, results)
                logger.debug("Reranking complete")
            
            final_results = results[:k]
            logger.info(f"Search complete, returning {len(final_results)} results")
            return final_results
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []
    
    def get_note_by_id(self, doc_id: str) -> Optional[T]:
        """Retrieve a note by its ID."""
        logger.debug(f"Retrieving note with ID: {doc_id}")
        note = self.notes.get(doc_id)
        if note:
            logger.debug(f"Successfully retrieved note with ID: {doc_id}")
        else:
            logger.warning(f"Note with ID {doc_id} not found")
        return note
    
    def save_index(self, path: str = None):
        """Save the FAISS index and associated data to disk."""
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
        """Load FAISS index and associated data from disk."""
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
        """Update an existing note in the index."""
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
        """Get all Zettel notes linked to a Cornell note."""
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
        """Get all Cornell notes that were generated from a specific source document."""
        notes_from_source = []
        for note_id, note in self.notes.items():
            if note.source_id == source_id:
                notes_from_source.append(note)
        
        return notes_from_source
    
    def update_source_info(self, note_id: str, source_id: str, source_title: str) -> bool:
        """Update the source information of a Cornell note."""
        note = self.get_note_by_id(note_id)
        if note:
            note.source_id = source_id
            note.source_title = source_title
            # Update the metadata dictionary as well
            if note_id in self.metadata_dict:
                self.metadata_dict[note_id]['source_id'] = source_id
                self.metadata_dict[note_id]['source_title'] = source_title
            return True
        return False
    
    def get_source_info(self, note_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Get the source ID and title of a Cornell note."""
        note = self.get_note_by_id(note_id)
        if note:
            return note.source_id, note.source_title
        return None, None


class ZettelNoteRetriever(FaissRetriever[ZettelNote]):
    """Specialized retriever for Zettel Notes"""
    def __init__(self, model_name: str = 'nomic-embed-text', index_path: Optional[str] = None):
        super().__init__(ZettelNote, model_name, index_path)
        
    def get_linked_notes(self, zettel_id: str) -> List[ZettelNote]:
        """Get all Zettel notes linked to a specific Zettel note."""
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
        """Get the parent Cornell note of a Zettel note."""
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