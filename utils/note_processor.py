from typing import List, Dict, Optional, Any, Tuple
from .cornell_zettel_memory_system import (
    CornellMethodNote, 
    ZettelNote, 
    generate_cornell_method_note, 
    get_Zettel_notes
)
from .retrievers import CornellNoteRetriever, ZettelNoteRetriever


class NoteProcessor:
    """
    Handles the process of creating Cornell and Zettel notes from text,
    finding similar notes, establishing links, and saving to indices.
    """
    
    def __init__(
        self, 
        cornell_retriever: Optional[CornellNoteRetriever] = None,
        zettel_retriever: Optional[ZettelNoteRetriever] = None
    ):
        # redacted
        self.cornell_retriever = cornell_retriever
        self.zettel_retriever = zettel_retriever
    
    def process_text(self, text: str, source_id: Optional[str] = None) -> Tuple[CornellMethodNote, List[ZettelNote]]:
        """
        Process text to create Cornell and Zettel notes.
        
        Args:
            text: The text to process
            source_id: Optional ID of the source document
            
        Returns:
            A tuple containing the Cornell note and a list of Zettel notes
        """
        # Create Cornell note
        cornell_note = self._create_cornell_note(text)
        
        # Set source_id if provided
        if source_id:
            cornell_note.source_id = source_id
        
        # Create Zettel notes
        zettel_notes = self._create_zettel_notes(cornell_note)
        
        # Establish links between notes
        for zettel_note in zettel_notes:
            self._establish_links(zettel_note)
        
        # Save notes to indices
        self._save_notes(cornell_note, zettel_notes)
        
        return cornell_note, zettel_notes
    
    def process_text_batch(self, texts: List[str], source_ids: Optional[List[str]] = None) -> List[Tuple[CornellMethodNote, List[ZettelNote]]]:
        """
        Process a batch of texts to create Cornell and Zettel notes.
        
        Args:
            texts: List of texts to process
            source_ids: Optional list of source document IDs corresponding to each text
            
        Returns:
            List of tuples containing Cornell notes and their associated Zettel notes
        """
        results = []
        
        for i, text in enumerate(texts):
            source_id = None
            if source_ids and i < len(source_ids):
                source_id = source_ids[i]
            
            cornell_note, zettel_notes = self.process_text(text, source_id)
            results.append((cornell_note, zettel_notes))
        
        return results
    
    def get_notes_by_source(self, source_id: str) -> List[Tuple[CornellMethodNote, List[ZettelNote]]]:
        """
        Retrieve all notes associated with a specific source document.
        
        Args:
            source_id: ID of the source document
            
        Returns:
            List of tuples containing Cornell notes and their associated Zettel notes
        """
        if not self.cornell_retriever:
            return []
        
        results = []
        cornell_notes = self.cornell_retriever.get_notes_by_source_id(source_id)
        
        for cornell_note in cornell_notes:
            zettel_notes = []
            if self.zettel_retriever:
                for zettel_id in cornell_note.zettle_ids:
                    zettel_note = self.zettel_retriever.get_note_by_id(zettel_id)
                    if zettel_note:
                        zettel_notes.append(zettel_note)
            
            results.append((cornell_note, zettel_notes))
        
        return results
    
    def _create_cornell_note(self, text: str) -> CornellMethodNote:
        """
        Create a Cornell note from the given text.
        
        Args:
            text: The text to create a Cornell note from
            
        Returns:
            A CornellMethodNote object
        """
        return generate_cornell_method_note(text)
    
    def _create_zettel_notes(self, cornell_note: CornellMethodNote) -> List[ZettelNote]:
        """
        Create Zettel notes from a Cornell note.
        
        Args:
            cornell_note: The Cornell note to create Zettel notes from
            
        Returns:
            A list of ZettelNote objects
        """
        return get_Zettel_notes(cornell_note)
    
    def _establish_links(self, zettel_note: ZettelNote) -> None:
        """
        Find similar Zettel notes and establish bidirectional links.
        
        Args:
            zettel_note: The new Zettel note to establish links for
        """
        if self.zettel_retriever is None:
            return
        
        # Search for similar notes using the content
        similar_notes = self.zettel_retriever.search(zettel_note.content, k=3)
        
        # Extract IDs of similar notes
        similar_ids = []
        for result in similar_notes:
            # Different retrievers might have different result structures
            if isinstance(result, dict) and 'id' in result:
                similar_ids.append(result['id'])
            elif hasattr(result, 'id'):
                similar_ids.append(result.id)
        
        # Add links to the new note
        zettel_note.links.extend(similar_ids)
        
        # Add bidirectional links to existing notes
        for note_id in similar_ids:
            existing_note = self.zettel_retriever.get_note_by_id(note_id)
            if existing_note and zettel_note.id not in existing_note.links:
                existing_note.links.append(zettel_note.id)
                # Update the existing note in the retriever
                self.zettel_retriever.update_note(existing_note)
    
    def _save_notes(self, cornell_note: CornellMethodNote, zettel_notes: List[ZettelNote]) -> None:
        """
        Save Cornell and Zettel notes to their respective indices.
        
        Args:
            cornell_note: The Cornell note to save
            zettel_notes: The Zettel notes to save
        """
        # Save Cornell note if retriever is available
        if self.cornell_retriever is not None:
            self.cornell_retriever.add_document(cornell_note)
        
        # Save Zettel notes if retriever is available
        if self.zettel_retriever is not None:
            for zettel_note in zettel_notes:
                self.zettel_retriever.add_document(zettel_note)