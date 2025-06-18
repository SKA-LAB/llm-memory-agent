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
        """
        Initialize the NoteProcessor with retrievers for Cornell and Zettel notes.
        
        Args:
            cornell_retriever: Retriever for Cornell notes
            zettel_retriever: Retriever for Zettel notes
        """
        self.cornell_retriever = cornell_retriever
        self.zettel_retriever = zettel_retriever
    
    def process_text(self, text: str) -> Tuple[CornellMethodNote, List[ZettelNote]]:
        """
        Process a block of text to create a Cornell note and associated Zettel notes,
        find similar notes, establish links, and save to indices.
        
        Args:
            text: The text to process
            
        Returns:
            A tuple containing the created Cornell note and list of Zettel notes
        """
        # Step 1: Create Cornell note from text
        cornell_note = generate_cornell_method_note(text)
        
        # Step 2: Generate Zettel notes from Cornell note
        zettel_notes = get_Zettel_notes(cornell_note)
        
        # Step 3: Find similar notes and establish links if retriever is available
        if self.zettel_retriever is not None:
            for zettel_note in zettel_notes:
                self._establish_links(zettel_note)
        
        # Step 4: Save notes to indices if retrievers are available
        self._save_notes(cornell_note, zettel_notes)
        
        return cornell_note, zettel_notes
    
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