import unittest
from unittest.mock import Mock, patch, MagicMock
import logging
from typing import List, Dict, Optional, Any, Tuple

from utils.note_processor import NoteProcessor
from utils.retrievers import CornellNoteRetriever, ZettelNoteRetriever
from utils.cornell_zettel_memory_system import CornellMethodNote, ZettelNote

class TestNoteProcessor(unittest.TestCase):
    def setUp(self):
        # Create mock retrievers
        self.mock_cornell_retriever = Mock(spec=CornellNoteRetriever)
        self.mock_zettel_retriever = Mock(spec=ZettelNoteRetriever)
        
        # Create the note processor with mock retrievers
        self.note_processor = NoteProcessor(
            cornell_retriever=self.mock_cornell_retriever,
            zettel_retriever=self.mock_zettel_retriever
        )
    
    def test_initialization(self):
        """Test that the NoteProcessor initializes correctly with retrievers"""
        self.assertEqual(self.note_processor.cornell_retriever, self.mock_cornell_retriever)
        self.assertEqual(self.note_processor.zettel_retriever, self.mock_zettel_retriever)
        
        # Test initialization with no retrievers
        empty_processor = NoteProcessor()
        self.assertIsNone(empty_processor.cornell_retriever)
        self.assertIsNone(empty_processor.zettel_retriever)
    
    @patch('utils.note_processor.generate_cornell_method_note')
    @patch('utils.note_processor.get_Zettel_notes')
    def test_process_text(self, mock_get_zettel_notes, mock_generate_cornell_note):
        """Test processing a single text into Cornell and Zettel notes"""
        # Setup mock returns
        mock_cornell_note = MagicMock(spec=CornellMethodNote)
        mock_cornell_note.id = "cornell-123"
        mock_cornell_note.title = "Test Cornell Note"
        mock_cornell_note.zettle_ids = []
        
        mock_zettel_note = MagicMock(spec=ZettelNote)
        mock_zettel_note.id = "zettel-123"
        mock_zettel_note.content = "Test Zettel content"
        mock_zettel_note.links = []
        
        mock_generate_cornell_note.return_value = mock_cornell_note
        mock_get_zettel_notes.return_value = [mock_zettel_note]
        
        # Mock the search method to return empty results (no similar notes)
        self.mock_zettel_retriever.search.return_value = []
        
        # Call the method
        result_cornell, result_zettels = self.note_processor.process_text(
            "Test text", 
            source_id="source-123", 
            source_title="Test Source"
        )
        
        # Verify results
        self.assertEqual(result_cornell, mock_cornell_note)
        self.assertEqual(result_zettels, [mock_zettel_note])
        
        # Verify source information was set
        self.assertEqual(mock_cornell_note.source_id, "source-123")
        self.assertEqual(mock_cornell_note.source_title, "Test Source")
        
        # Verify methods were called
        mock_generate_cornell_note.assert_called_once_with("Test text")
        mock_get_zettel_notes.assert_called_once_with(mock_cornell_note)
        self.mock_cornell_retriever.add_document.assert_called_once_with(mock_cornell_note)
        self.mock_zettel_retriever.add_document.assert_called_once_with(mock_zettel_note)
    
    @patch('utils.note_processor.generate_cornell_method_note')
    @patch('utils.note_processor.get_Zettel_notes')
    def test_process_text_batch(self, mock_get_zettel_notes, mock_generate_cornell_note):
        """Test processing multiple texts into Cornell and Zettel notes"""
        # Setup mock returns for two notes
        mock_cornell_notes = [
            MagicMock(spec=CornellMethodNote, id=f"cornell-{i}", title=f"Test Cornell Note {i}", zettle_ids=[])
            for i in range(2)
        ]
        
        # Fix: Create mock Zettel notes without using undefined variable j
        mock_zettel_notes = [
            [MagicMock(spec=ZettelNote, id=f"zettel-{i}-0", content=f"Test Zettel content {i}", links=[])]
            for i in range(2)
        ]
        
        mock_generate_cornell_note.side_effect = mock_cornell_notes
        mock_get_zettel_notes.side_effect = mock_zettel_notes
        
        # Mock the search method to return empty results
        self.mock_zettel_retriever.search.return_value = []
        
        # Call the method
        results = self.note_processor.process_text_batch(
            ["Text 1", "Text 2"],
            source_ids=["source-1", "source-2"],
            source_titles=["Source 1", "Source 2"]
        )
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], mock_cornell_notes[0])
        self.assertEqual(results[0][1], mock_zettel_notes[0])
        self.assertEqual(results[1][0], mock_cornell_notes[1])
        self.assertEqual(results[1][1], mock_zettel_notes[1])
        
        # Verify source information was set
        self.assertEqual(mock_cornell_notes[0].source_id, "source-1")
        self.assertEqual(mock_cornell_notes[0].source_title, "Source 1")
        self.assertEqual(mock_cornell_notes[1].source_id, "source-2")
        self.assertEqual(mock_cornell_notes[1].source_title, "Source 2")
    
    def test_get_notes_by_source(self):
        """Test retrieving notes by source ID"""
        # Setup mock Cornell notes
        mock_cornell_notes = [
            MagicMock(spec=CornellMethodNote, id=f"cornell-{i}", zettle_ids=[f"zettel-{i}-{j}" for j in range(2)])
            for i in range(2)
        ]
        
        # Setup mock Zettel notes
        mock_zettel_notes = {
            f"zettel-{i}-{j}": MagicMock(spec=ZettelNote, id=f"zettel-{i}-{j}")
            for i in range(2) for j in range(2)
        }
        
        # Configure mock retrievers
        self.mock_cornell_retriever.get_notes_by_source_id.return_value = mock_cornell_notes
        self.mock_zettel_retriever.get_note_by_id.side_effect = lambda id: mock_zettel_notes.get(id)
        
        # Call the method
        results = self.note_processor.get_notes_by_source("source-123")
        
        # Verify results
        self.assertEqual(len(results), 2)
        for i, (cornell_note, zettel_notes) in enumerate(results):
            self.assertEqual(cornell_note, mock_cornell_notes[i])
            self.assertEqual(len(zettel_notes), 2)
            for j, zettel_note in enumerate(zettel_notes):
                self.assertEqual(zettel_note, mock_zettel_notes[f"zettel-{i}-{j}"])
        
        # Verify method was called with correct source ID
        self.mock_cornell_retriever.get_notes_by_source_id.assert_called_once_with("source-123")
    
    @patch('utils.note_processor.generate_cornell_method_note')
    @patch('utils.note_processor.get_Zettel_notes')
    def test_establish_links(self, mock_get_zettel_notes, mock_generate_cornell_note):
        """Test establishing links between Zettel notes"""
        # Setup mock notes
        mock_cornell_note = MagicMock(spec=CornellMethodNote, id="cornell-123", zettle_ids=[])
        
        new_zettel_note = MagicMock(spec=ZettelNote, id="new-zettel", content="New content", links=[])
        
        existing_zettel_notes = [
            MagicMock(spec=ZettelNote, id=f"existing-{i}", links=[]) for i in range(3)
        ]
        
        # Configure mocks
        mock_generate_cornell_note.return_value = mock_cornell_note
        mock_get_zettel_notes.return_value = [new_zettel_note]
        
        # Mock search to return similar notes
        self.mock_zettel_retriever.search.return_value = [
            {"id": note.id} for note in existing_zettel_notes
        ]
        
        # Mock get_note_by_id to return existing notes
        self.mock_zettel_retriever.get_note_by_id.side_effect = lambda id: next(
            (note for note in existing_zettel_notes if note.id == id), None
        )
        
        # Call process_text which will trigger _establish_links
        self.note_processor.process_text("Test text")
        
        # Verify links were established
        self.assertEqual(len(new_zettel_note.links), 3)
        for i, note in enumerate(existing_zettel_notes):
            self.assertIn(note.id, new_zettel_note.links)
            self.assertIn(new_zettel_note.id, note.links)
            self.mock_zettel_retriever.update_note.assert_any_call(note)
    
    def test_save_notes_with_no_retrievers(self):
        """Test saving notes when retrievers are not available"""
        # Create processor with no retrievers
        processor = NoteProcessor()
        
        # Create mock notes with necessary attributes
        mock_cornell_note = MagicMock(spec=CornellMethodNote)
        mock_cornell_note.id = "mock-cornell-id"  # Add id attribute
        
        mock_zettel_note = MagicMock(spec=ZettelNote)
        mock_zettel_note.id = "mock-zettel-id"  # Add id attribute
        mock_zettel_notes = [mock_zettel_note]
        
        # This should not raise an exception
        processor._save_notes(mock_cornell_note, mock_zettel_notes)
        
        # No methods should have been called on the retrievers since they don't exist
        # We can't assert on mock_cornell_note.assert_not_called() because we're using it as a data object
        # Instead, we can verify that the processor's retrievers are None
        self.assertIsNone(processor.cornell_retriever)
        self.assertIsNone(processor.zettel_retriever)

if __name__ == '__main__':
    unittest.main()