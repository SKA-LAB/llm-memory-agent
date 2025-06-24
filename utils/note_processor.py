from typing import List, Dict, Optional, Any, Tuple
import logging
from utils.cornell_zettel_memory_system import (
    CornellMethodNote, 
    ZettelNote, 
    generate_cornell_method_note, 
    get_Zettel_notes
)
from utils.retrievers import CornellNoteRetriever, ZettelNoteRetriever

# Set up logger for this module
logger = logging.getLogger(__name__)

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
        logger.info("Initializing NoteProcessor")
        self.cornell_retriever = cornell_retriever
        self.zettel_retriever = zettel_retriever
        logger.debug(f"Cornell retriever initialized: {cornell_retriever is not None}")
        logger.debug(f"Zettel retriever initialized: {zettel_retriever is not None}")
    
    def process_text(self, text: str, source_id: Optional[str] = None, source_title: Optional[str] = None) -> Tuple[CornellMethodNote, List[ZettelNote]]:
        """
        Process a single text into Cornell and Zettel notes.
        
        Args:
            text: The text to process
            source_id: Optional ID of the source document
            source_title: Optional title of the source document
            
        Returns:
            A tuple containing the Cornell note and a list of Zettel notes
        """
        logger.info(f"Processing text (source_id: {source_id}, source_title: {source_title})")
        logger.debug(f"Text length: {len(text)} characters")
        
        # Create Cornell note
        cornell_note = self._create_cornell_note(text)
        logger.info(f"Created Cornell note with ID: {cornell_note.id}")
        
        # Set source information if provided
        if source_id:
            cornell_note.source_id = source_id
            logger.debug(f"Set source_id: {source_id}")
        if source_title:
            cornell_note.source_title = source_title
            logger.debug(f"Set source_title: {source_title}")
            
        # Create Zettel notes
        zettel_notes = self._create_zettel_notes(cornell_note)
        logger.info(f"Created {len(zettel_notes)} Zettel notes")
        
        # Establish links between Zettel notes
        for zettel_note in zettel_notes:
            self._establish_links(zettel_note)
            
        # Save notes to indices
        self._save_notes(cornell_note, zettel_notes)
        
        return cornell_note, zettel_notes
    
    def process_text_batch(self, texts: List[str], source_ids: Optional[List[str]] = None, source_titles: Optional[List[str]] = None) -> List[Tuple[CornellMethodNote, List[ZettelNote]]]:
        """
        Process multiple texts into Cornell and Zettel notes.
        
        Args:
            texts: List of texts to process
            source_ids: Optional list of source document IDs
            source_titles: Optional list of source document titles
            
        Returns:
            List of tuples containing Cornell notes and their associated Zettel notes
        """
        logger.info(f"Processing batch of {len(texts)} texts")
        results = []
        
        for i, text in enumerate(texts):
            source_id = source_ids[i] if source_ids and i < len(source_ids) else None
            source_title = source_titles[i] if source_titles and i < len(source_titles) else None
            logger.debug(f"Processing batch item {i+1}/{len(texts)}")
            cornell_note, zettel_notes = self.process_text(text, source_id, source_title)
            results.append((cornell_note, zettel_notes))
            
        logger.info(f"Completed batch processing of {len(texts)} texts")
        return results
    
    def get_notes_by_source(self, source_id: str) -> List[Tuple[CornellMethodNote, List[ZettelNote]]]:
        """
        Retrieve all notes associated with a specific source document.
        
        Args:
            source_id: ID of the source document
            
        Returns:
            List of tuples containing Cornell notes and their associated Zettel notes
        """
        logger.info(f"Retrieving notes for source_id: {source_id}")
        
        if not self.cornell_retriever:
            logger.warning("Cornell retriever not available, returning empty result")
            return []
        
        results = []
        cornell_notes = self.cornell_retriever.get_notes_by_source_id(source_id)
        logger.info(f"Found {len(cornell_notes)} Cornell notes for source_id: {source_id}")
        
        for cornell_note in cornell_notes:
            zettel_notes = []
            if self.zettel_retriever:
                for zettel_id in cornell_note.zettle_ids:
                    zettel_note = self.zettel_retriever.get_note_by_id(zettel_id)
                    if zettel_note:
                        zettel_notes.append(zettel_note)
                    else:
                        logger.warning(f"Referenced Zettel note with ID {zettel_id} not found")
            else:
                logger.debug("Zettel retriever not available, skipping Zettel note retrieval")
            
            results.append((cornell_note, zettel_notes))
            logger.debug(f"Added Cornell note {cornell_note.id} with {len(zettel_notes)} Zettel notes to results")
        
        return results
    
    def _create_cornell_note(self, text: str) -> CornellMethodNote:
        """
        Create a Cornell note from the given text.
        
        Args:
            text: The text to create a Cornell note from
            
        Returns:
            A CornellMethodNote object
        """
        logger.debug("Creating Cornell note from text")
        cornell_note = generate_cornell_method_note(text)
        logger.debug(f"Created Cornell note with title: {cornell_note.title}")
        return cornell_note
    
    def _create_zettel_notes(self, cornell_note: CornellMethodNote) -> List[ZettelNote]:
        """
        Create Zettel notes from a Cornell note.
        
        Args:
            cornell_note: The Cornell note to create Zettel notes from
            
        Returns:
            A list of ZettelNote objects
        """
        logger.debug(f"Creating Zettel notes from Cornell note: {cornell_note.id}")
        zettel_notes = get_Zettel_notes(cornell_note)
        logger.debug(f"Created {len(zettel_notes)} Zettel notes")
        return zettel_notes
    
    def _establish_links(self, zettel_note: ZettelNote) -> None:
        """
        Find similar Zettel notes and establish bidirectional links.
        
        Args:
            zettel_note: The new Zettel note to establish links for
        """
        logger.debug(f"Establishing links for Zettel note: {zettel_note.id}")
        
        if self.zettel_retriever is None:
            logger.warning("Zettel retriever not available, skipping link establishment")
            return
        
        # Search for similar notes using the content
        similar_notes = self.zettel_retriever.search(zettel_note.content, k=3)
        logger.debug(f"Found {len(similar_notes)} similar notes")
        
        # Extract IDs of similar notes
        similar_ids = []
        for result in similar_notes:
            # Different retrievers might have different result structures
            if isinstance(result, dict) and 'id' in result:
                similar_ids.append(result['id'])
            elif hasattr(result, 'id'):
                similar_ids.append(result.id)
        
        logger.debug(f"Similar note IDs: {similar_ids}")
        
        # Add links to the new note
        zettel_note.links.extend(similar_ids)
        
        # Add bidirectional links to existing notes
        for note_id in similar_ids:
            existing_note = self.zettel_retriever.get_note_by_id(note_id)
            if existing_note and zettel_note.id not in existing_note.links:
                existing_note.links.append(zettel_note.id)
                logger.debug(f"Added bidirectional link between {zettel_note.id} and {note_id}")
                # Update the existing note in the retriever
                self.zettel_retriever.update_note(existing_note)
            elif existing_note:
                logger.debug(f"Link already exists between {zettel_note.id} and {note_id}")
            else:
                logger.warning(f"Could not find note with ID {note_id} for linking")
    
    def _save_notes(self, cornell_note: CornellMethodNote, zettel_notes: List[ZettelNote]) -> None:
        """
        Save Cornell and Zettel notes to their respective indices.
        
        Args:
            cornell_note: The Cornell note to save
            zettel_notes: The Zettel notes to save
        """
        logger.debug(f"Saving Cornell note {cornell_note.id} and {len(zettel_notes)} Zettel notes")
        
        # Save Cornell note if retriever is available
        if self.cornell_retriever is not None:
            self.cornell_retriever.add_document(cornell_note)
            logger.debug(f"Saved Cornell note {cornell_note.id} to index")
        else:
            logger.warning("Cornell retriever not available, Cornell note not saved")
        
        # Save Zettel notes if retriever is available
        if self.zettel_retriever is not None:
            for zettel_note in zettel_notes:
                self.zettel_retriever.add_document(zettel_note)
                logger.debug(f"Saved Zettel note {zettel_note.id} to index")
        else:
            logger.warning("Zettel retriever not available, Zettel notes not saved")
