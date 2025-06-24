from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from datetime import datetime
import uuid
import os
import json
import logging

from utils.retrievers import CornellNoteRetriever, ZettelNoteRetriever

logger = logging.getLogger(__name__)

class SourceDocument(BaseModel):
    """Represents a source document from which notes are generated."""
    id: str = uuid.uuid4().hex
    title: str
    author: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None
    content: str
    metadata: Dict[str, Any] = {}
    created_at: str = datetime.now().isoformat()

class SourceDocumentManager:
    """Manages source documents and their relationships with notes."""
    
    def __init__(
        self,
        cornell_retriever: Optional[CornellNoteRetriever] = None,
        zettel_retriever: Optional[ZettelNoteRetriever] = None,
        storage_path: Optional[str] = None
    ):
        self.cornell_retriever = cornell_retriever
        self.zettel_retriever = zettel_retriever
        
        # Set default storage path if None is provided
        if storage_path is None:
            home_dir = os.path.expanduser("~")
            default_dir = os.path.join(home_dir, ".llm_memory_agent", "source_documents")
            os.makedirs(default_dir, exist_ok=True)
            self.storage_path = default_dir
        else:
            self.storage_path = storage_path
            os.makedirs(storage_path, exist_ok=True)
        
        # Load existing documents
        self.documents = {}
        self._load_documents()
    
    def _load_documents(self):
        """Load source documents from storage."""
        try:
            documents_file = os.path.join(self.storage_path, "source_documents.json")
            if os.path.exists(documents_file):
                with open(documents_file, 'r') as f:
                    documents_data = json.load(f)
                    for doc_data in documents_data:
                        document = SourceDocument(**doc_data)
                        self.documents[document.id] = document
                logger.info(f"Loaded {len(self.documents)} source documents")
        except Exception as e:
            logger.error(f"Error loading source documents: {e}")
    
    def _save_documents(self):
        """Save source documents to storage."""
        try:
            documents_file = os.path.join(self.storage_path, "source_documents.json")
            with open(documents_file, 'w') as f:
                documents_data = [doc.model_dump() for doc in self.documents.values()]
                json.dump(documents_data, f, indent=2)
            logger.info(f"Saved {len(self.documents)} source documents")
        except Exception as e:
            logger.error(f"Error saving source documents: {e}")
    
    def add_document(self, document: SourceDocument) -> str:
        """Add a source document.
        
        Args:
            document: The source document to add
            
        Returns:
            The ID of the added document
        """
        self.documents[document.id] = document
        self._save_documents()
        return document.id
    
    def get_document(self, doc_id: str) -> Optional[SourceDocument]:
        """Get a source document by ID.
        
        Args:
            doc_id: ID of the document to retrieve
            
        Returns:
            The source document if found, None otherwise
        """
        return self.documents.get(doc_id)
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a source document.
        
        Args:
            doc_id: ID of the document to delete
            
        Returns:
            Whether the deletion was successful
        """
        if doc_id in self.documents:
            del self.documents[doc_id]
            self._save_documents()
            return True
        return False
    
    def get_related_notes(self, doc_id: str) -> Dict[str, List]:
        """Get all notes related to a source document.
        
        Args:
            doc_id: ID of the source document
            
        Returns:
            Dictionary with cornell_notes and zettel_notes lists
        """
        result = {
            "cornell_notes": [],
            "zettel_notes": []
        }
        
        if not self.cornell_retriever:
            return result
        
        # Get Cornell notes from the source
        cornell_notes = self.cornell_retriever.get_notes_by_source_id(doc_id)
        result["cornell_notes"] = cornell_notes
        
        # Get related Zettel notes
        if self.zettel_retriever:
            zettel_notes = []
            for cornell_note in cornell_notes:
                for zettel_id in cornell_note.zettle_ids:
                    zettel_note = self.zettel_retriever.get_note_by_id(zettel_id)
                    if zettel_note:
                        zettel_notes.append(zettel_note)
            result["zettel_notes"] = zettel_notes
        
        return result
    
    def create_document_from_text(
        self, 
        title: str, 
        content: str, 
        author: Optional[str] = None,
        url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SourceDocument:
        """Create a new source document from text.
        
        Args:
            title: Title of the document
            content: Content of the document
            author: Optional author of the document
            url: Optional URL of the document
            metadata: Optional additional metadata
            
        Returns:
            The created source document
        """
        document = SourceDocument(
            title=title,
            content=content,
            author=author,
            url=url,
            date=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        self.add_document(document)
        return document