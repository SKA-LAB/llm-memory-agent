import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import tempfile
import numpy as np
import faiss
import pickle
from datetime import datetime

from utils.retrievers import FaissRetriever, CornellNoteRetriever, ZettelNoteRetriever
from utils.cornell_zettel_memory_system import CornellMethodNote, ZettelNote, ZettelSimple, CornellSimple
from utils.search_reranker import SearchReranker

class TestFaissRetriever(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.index_path = os.path.join(self.test_dir, "test_index")
        
        # Mock the OllamaEmbeddings class
        self.mock_embeddings_patcher = patch('utils.retrievers.OllamaEmbeddings')
        self.mock_embeddings_class = self.mock_embeddings_patcher.start()
        self.mock_embeddings = self.mock_embeddings_class.return_value
        
        # Configure the mock embeddings to return a fixed embedding
        self.test_embedding_dim = 384
        self.mock_embeddings.embed_query.return_value = [0.1] * self.test_embedding_dim
        
        # Create a sample ZettelNote for testing
        self.sample_note = ZettelNote(
            id="test-note-1",
            content="This is a test note for testing the retriever.",
            type="standard",
            links=[],
            tags=["test", "retriever"],
            created_at=datetime.now().isoformat(),
            accessed_at=datetime.now().isoformat(),
            retrieval_count=0,
            cornell_id="some-id",
            metadata={},
            note_simple=ZettelSimple(title="Test Note", keywords=["test", "retriever"], body="This is a test note."),
        )
        
        # Initialize the retriever with the mock embeddings
        self.retriever = FaissRetriever(
            note_class=ZettelNote,
            model_name='test-model',
            index_path=self.index_path,
            use_reranker=False  # Disable reranker for basic tests
        )
        
        # Replace the embeddings with our mock
        self.retriever.embeddings = self.mock_embeddings
        self.retriever.embedding_dim = self.test_embedding_dim
    
    def tearDown(self):
        """Clean up after each test method."""
        self.mock_embeddings_patcher.stop()
        
        # Clean up temporary files
        if os.path.exists(f"{self.index_path}.index"):
            os.remove(f"{self.index_path}.index")
        if os.path.exists(f"{self.index_path}.data"):
            os.remove(f"{self.index_path}.data")
    
    def test_initialization(self):
        """Test that the retriever initializes correctly."""
        self.assertEqual(self.retriever.note_class, ZettelNote)
        self.assertEqual(self.retriever.class_name, "ZettelNote")
        self.assertEqual(self.retriever.embedding_dim, self.test_embedding_dim)
        self.assertEqual(self.retriever.index_path, self.index_path)
        self.assertFalse(self.retriever.use_reranker)
        self.assertIsNone(self.retriever.reranker)
        self.assertEqual(len(self.retriever.notes), 0)
        self.assertEqual(len(self.retriever.note_ids), 0)
    
    def test_add_document(self):
        """Test adding a document to the retriever."""
        doc_id = self.retriever.add_document(self.sample_note)
        
        # Check that the document was added correctly
        self.assertEqual(doc_id, self.sample_note.id)
        self.assertIn(doc_id, self.retriever.notes)
        self.assertIn(doc_id, self.retriever.note_ids)
        self.assertIn(doc_id, self.retriever.metadata_dict)
        
        # Check that the embedding was generated and added to the index
        self.mock_embeddings.embed_query.assert_called_with(self.sample_note.content)
        self.assertEqual(self.retriever.index.ntotal, 1)
    
    def test_get_note_by_id(self):
        """Test retrieving a note by ID."""
        # Add a document first
        self.retriever.add_document(self.sample_note)
        
        # Get existing note
        note = self.retriever.get_note_by_id(self.sample_note.id)
        self.assertIsNotNone(note)
        self.assertEqual(note.id, self.sample_note.id)
        self.assertEqual(note.title, self.sample_note.title)
        
        # Get non-existent note
        note = self.retriever.get_note_by_id("nonexistent")
        self.assertIsNone(note)
    
    def test_delete_document(self):
        """Test deleting a document from the retriever."""
        # Add a document first
        self.retriever.add_document(self.sample_note)
        
        # Delete the document
        result = self.retriever.delete_document(self.sample_note.id)
        self.assertTrue(result)
        
        # Check that the document was marked as deleted
        self.assertNotIn(self.sample_note.id, self.retriever.notes)
        self.assertNotIn(self.sample_note.id, self.retriever.metadata_dict)
        
        # Try to delete a non-existent document
        result = self.retriever.delete_document("nonexistent")
        self.assertFalse(result)
    
    @patch('utils.retrievers.faiss.IndexFlatL2')
    def test_rebuild_index(self, mock_index_class):
        """Test rebuilding the index after deletions."""
        # Setup mock index
        mock_index = MagicMock()
        mock_index_class.return_value = mock_index
        self.retriever.index = mock_index
        
        # Add multiple documents
        for i in range(5):
            note = ZettelNote(
                id=f"test-note-{i}",
                content=f"This is test note {i}",
                type="standard",
                links=[],
                tags=["test"],
                created_at=datetime.now().isoformat(),
                accessed_at=datetime.now().isoformat(),
                retrieval_count=0,
                cornell_id="some-id",
                metadata={},
                note_simple=ZettelSimple(title=f"Test Note {i}", keywords=["test"], body=f"This is test note {i}."),
            )
            self.retriever.add_document(note)
        
        # Delete some documents
        self.retriever.delete_document("test-note-1")
        self.retriever.delete_document("test-note-3")
        
        # Manually trigger rebuild
        self.retriever._rebuild_index()
        
        # Check that the index was rebuilt with only valid documents
        self.assertEqual(len(self.retriever.note_ids), 3)
        self.assertIn("test-note-0", self.retriever.note_ids)
        self.assertIn("test-note-2", self.retriever.note_ids)
        self.assertIn("test-note-4", self.retriever.note_ids)
        self.assertNotIn("test-note-1", self.retriever.note_ids)
        self.assertNotIn("test-note-3", self.retriever.note_ids)
        self.assertNotIn(None, self.retriever.note_ids)
    
    def test_search(self):
        """Test searching for similar notes."""
        # Add multiple documents
        for i in range(5):
            note = ZettelNote(
                id=f"test-note-{i}",
                content=f"This is test note {i}",
                type="standard",
                links=[],
                tags=["test"],
                created_at=datetime.now().isoformat(),
                accessed_at=datetime.now().isoformat(),
                retrieval_count=0,
                cornell_id="some_id",
                metadata={},
                note_simple=ZettelSimple(title=f"Test Note {i}", keywords=["test"], body=f"This is test note {i}."),
            )
            self.retriever.add_document(note)
        
        # Mock the FAISS search method to return fixed indices and distances
        self.retriever.index.search = MagicMock(return_value=(
            np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),  # Distances
            np.array([[0, 1, 2, 3, 4]])  # Indices
        ))
        
        # Search for similar notes
        results = self.retriever.search("test query", k=3)
        
        # Check that the search returned the expected number of results
        self.assertEqual(len(results), 3)
        
        # Check that the results contain the expected notes
        self.assertEqual(results[0]['id'], "test-note-0")
        self.assertEqual(results[1]['id'], "test-note-1")
        self.assertEqual(results[2]['id'], "test-note-2")
        
        # Check that the scores were calculated correctly
        self.assertAlmostEqual(results[0]['score'], 1.0 / 1.1)
        self.assertAlmostEqual(results[1]['score'], 1.0 / 1.2)
        self.assertAlmostEqual(results[2]['score'], 1.0 / 1.3)
    
    def test_update_note(self):
        """Test updating an existing note."""
        # Add a document first
        self.retriever.add_document(self.sample_note)
        
        # Update the note
        updated_note = ZettelNote(
            id=self.sample_note.id,
            content=self.sample_note.content,  # Content remains the same
            type="standard",
            links=["link-1", "link-2"],
            tags=["test", "updated"],
            created_at=self.sample_note.created_at,
            accessed_at=datetime.now().isoformat(),
            retrieval_count=1,
            cornell_id="cornell-1",
            metadata={"key": "value"},
            note_simple=ZettelSimple(
                title="Updated Test Note",
                keywords=["test", "updated"],
                body="This note was updated."
            )
        )
        
        result = self.retriever.update_note(updated_note)
        self.assertTrue(result)
        
        # Check that the note was updated
        stored_note = self.retriever.get_note_by_id(self.sample_note.id)
        self.assertEqual(stored_note.title, "Updated Test Note")
        self.assertEqual(stored_note.links, ["link-1", "link-2"])
        self.assertEqual(stored_note.tags, ["test", "updated"])
        self.assertEqual(stored_note.retrieval_count, 1)
        self.assertEqual(stored_note.cornell_id, "cornell-1")
        self.assertEqual(stored_note.metadata, {"key": "value"})
        
        # Try to update a non-existent note
        non_existent_note = ZettelNote(
            id="nonexistent",
            content="This note doesn't exist in the index",
            type="standard",
            links=[],
            tags=[],
            created_at=datetime.now().isoformat(),
            accessed_at=datetime.now().isoformat(),
            retrieval_count=0,
            cornell_id="some-id",
            metadata={},
            note_simple=ZettelSimple(title="Nonexistent Note", keywords=[], body="This note doesn't exist.")
        )
        
        result = self.retriever.update_note(non_existent_note)
        self.assertFalse(result)
    
    def test_save_and_load_index(self):
        """Test saving and loading the index."""
        # Add some documents
        for i in range(3):
            note = ZettelNote(
                id=f"test-note-{i}",
                content=f"This is test note {i}",
                type="standard",
                links=[],
                tags=["test"],
                created_at=datetime.now().isoformat(),
                accessed_at=datetime.now().isoformat(),
                retrieval_count=0,
                cornell_id="some-id",
                metadata={},
                note_simple=ZettelSimple(title=f"Test Note {i}", keywords=["test"], body=f"This is test note {i}."),
            )
            self.retriever.add_document(note)
        
        # Save the index
        with patch('utils.retrievers.faiss.write_index') as mock_write_index, \
             patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            self.retriever.save_index()
            mock_write_index.assert_called_once()
            mock_file.assert_called_once()
        
        # Create a new retriever and load the index
        with patch('utils.retrievers.faiss.read_index') as mock_read_index, \
             patch('builtins.open', unittest.mock.mock_open(read_data=pickle.dumps({
                'notes': self.retriever.notes,
                'note_ids': self.retriever.note_ids,
                'metadata': self.retriever.metadata_dict,
                'class_name': self.retriever.class_name
             }))) as mock_file:
            new_retriever = FaissRetriever(
                note_class=ZettelNote,
                model_name='test-model',
                index_path=self.index_path,
                use_reranker=False
            )
            
            # Check that the data was loaded correctly
            self.assertEqual(len(new_retriever.notes), 3)
            self.assertEqual(len(new_retriever.note_ids), 3)
            self.assertEqual(len(new_retriever.metadata_dict), 3)


class TestCornellNoteRetriever(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.index_path = os.path.join(self.test_dir, "test_cornell_index")
        
        # Mock the OllamaEmbeddings class
        self.mock_embeddings_patcher = patch('utils.retrievers.OllamaEmbeddings')
        self.mock_embeddings_class = self.mock_embeddings_patcher.start()
        self.mock_embeddings = self.mock_embeddings_class.return_value
        
        # Configure the mock embeddings to return a fixed embedding
        self.test_embedding_dim = 384
        self.mock_embeddings.embed_query.return_value = [0.1] * self.test_embedding_dim
        
        # Initialize the Cornell retriever
        self.cornell_retriever = CornellNoteRetriever(
            model_name='test-model',
            index_path=self.index_path
        )
        
        # Replace the embeddings with our mock
        self.cornell_retriever.embeddings = self.mock_embeddings
        self.cornell_retriever.embedding_dim = self.test_embedding_dim
        
        # Create a sample Cornell note for testing
        self.sample_cornell_note = CornellMethodNote(
            id="cornell-test-1",
            source_id="source-123",
            source_title="Test Source",
            zettle_ids=[],
            created_at=datetime.now().isoformat(),
            accessed_at=datetime.now().isoformat(),
            retrieval_count=0,
            content="Test content",
            note_simple=CornellSimple(main_note="Test main note", questions=["What is this note?"], summary="Test summary")
        )
    
    def tearDown(self):
        """Clean up after each test method."""
        self.mock_embeddings_patcher.stop()
        
        # Clean up temporary files
        if os.path.exists(f"{self.index_path}.index"):
            os.remove(f"{self.index_path}.index")
        if os.path.exists(f"{self.index_path}.data"):
            os.remove(f"{self.index_path}.data")
    
    def test_initialization(self):
        """Test that the Cornell note retriever initializes correctly."""
        self.assertEqual(self.cornell_retriever.note_class, CornellMethodNote)
        self.assertEqual(self.cornell_retriever.class_name, "CornellMethodNote")
        self.assertEqual(self.cornell_retriever.embedding_dim, self.test_embedding_dim)
        self.assertEqual(self.cornell_retriever.index_path, self.index_path)
    
    def test_add_document(self):
        """Test adding a Cornell note to the retriever."""
        doc_id = self.cornell_retriever.add_document(self.sample_cornell_note)
        
        # Check that the document was added correctly
        self.assertEqual(doc_id, self.sample_cornell_note.id)
        self.assertIn(doc_id, self.cornell_retriever.notes)
        self.assertIn(doc_id, self.cornell_retriever.note_ids)
        self.assertIn(doc_id, self.cornell_retriever.metadata_dict)
        
        # Check that the embedding was generated and added to the index
        expected_content = f"{self.sample_cornell_note.content}"
        self.mock_embeddings.embed_query.assert_called_with(expected_content)
        self.assertEqual(self.cornell_retriever.index.ntotal, 1)
    
    def test_get_notes_by_source_id(self):
        """Test retrieving Cornell notes by source ID."""
        # Add multiple notes with different source IDs
        source_ids = ["source-1", "source-2", "source-1", "source-3"]
        for i, source_id in enumerate(source_ids):
            note = CornellMethodNote(
                id=f"cornell-{i}",
                source_id=source_id,
                source_title=f"Source {i}",
                zettle_ids=[],
                created_at=datetime.now().isoformat(),
                accessed_at=datetime.now().isoformat(),
                retrieval_count=0,
                content=f"Test content {i}",
                note_simple=CornellSimple(main_note=f"Test main note {i}", questions=[f"What is this note {i}?"], summary=f"Test summary {i}")
            )
            self.cornell_retriever.add_document(note)
        
        # Get notes for source-1
        source_1_notes = self.cornell_retriever.get_notes_by_source_id("source-1")
        self.assertEqual(len(source_1_notes), 2)
        self.assertEqual(source_1_notes[0].id, "cornell-0")
        self.assertEqual(source_1_notes[1].id, "cornell-2")
        
        # Get notes for source-2
        source_2_notes = self.cornell_retriever.get_notes_by_source_id("source-2")
        self.assertEqual(len(source_2_notes), 1)
        self.assertEqual(source_2_notes[0].id, "cornell-1")
        
        # Get notes for non-existent source
        non_existent_notes = self.cornell_retriever.get_notes_by_source_id("non-existent")
        self.assertEqual(len(non_existent_notes), 0)


class TestZettelNoteRetriever(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.index_path = os.path.join(self.test_dir, "test_zettel_index")
        
        # Mock the OllamaEmbeddings class
        self.mock_embeddings_patcher = patch('utils.retrievers.OllamaEmbeddings')
        self.mock_embeddings_class = self.mock_embeddings_patcher.start()
        self.mock_embeddings = self.mock_embeddings_class.return_value
        
        # Configure the mock embeddings to return a fixed embedding
        self.test_embedding_dim = 384
        self.mock_embeddings.embed_query.return_value = [0.1] * self.test_embedding_dim
        
        # Initialize the Zettel retriever
        self.zettel_retriever = ZettelNoteRetriever(
            model_name='test-model',
            index_path=self.index_path
        )
        
        # Replace the embeddings with our mock
        self.zettel_retriever.embeddings = self.mock_embeddings
        self.zettel_retriever.embedding_dim = self.test_embedding_dim
        
        # Create a sample Zettel note for testing
        self.sample_zettel_note = ZettelNote(
            id="zettel-test-1",
            content="This is a test Zettel note for testing the retriever.",
            type="standard",
            links=[],
            tags=["test", "zettel"],
            created_at=datetime.now().isoformat(),
            accessed_at=datetime.now().isoformat(),
            retrieval_count=0,
            cornell_id="cornell-1",
            metadata={},
            note_simple=ZettelSimple(title="Test Zettel", body="Test content", keywords=["test", "zettel"])
        )
    
    def tearDown(self):
        """Clean up after each test method."""
        self.mock_embeddings_patcher.stop()
        
        # Clean up temporary files
        if os.path.exists(f"{self.index_path}.index"):
            os.remove(f"{self.index_path}.index")
        if os.path.exists(f"{self.index_path}.data"):
            os.remove(f"{self.index_path}.data")
    
    def test_initialization(self):
        """Test that the Zettel note retriever initializes correctly."""
        self.assertEqual(self.zettel_retriever.note_class, ZettelNote)
        self.assertEqual(self.zettel_retriever.class_name, "ZettelNote")
        self.assertEqual(self.zettel_retriever.embedding_dim, self.test_embedding_dim)
        self.assertEqual(self.zettel_retriever.index_path, self.index_path)
    
    def test_add_document(self):
        """Test adding a Zettel note to the retriever."""
        doc_id = self.zettel_retriever.add_document(self.sample_zettel_note)
        
        # Check that the document was added correctly
        self.assertEqual(doc_id, self.sample_zettel_note.id)
        self.assertIn(doc_id, self.zettel_retriever.notes)
        self.assertIn(doc_id, self.zettel_retriever.note_ids)
        self.assertIn(doc_id, self.zettel_retriever.metadata_dict)
        
        # Check that the embedding was generated and added to the index
        self.mock_embeddings.embed_query.assert_called_with(self.sample_zettel_note.content)
        self.assertEqual(self.zettel_retriever.index.ntotal, 1)
        
        # Create a sample Cornell note for testing
        self.sample_cornell_note = CornellMethodNote(
            id="cornell-test-1",
            source_id="source-123",
            source_title="Test Source",
            zettle_ids=[],
            created_at=datetime.now().isoformat(),
            accessed_at=datetime.now().isoformat(),
            retrieval_count=0,
            content="Test content",
            note_simple=CornellSimple(main_note="Test Main Note", questions=["what is a test?"], summary="Test Summary")
        )
    
    def tearDown(self):
        """Clean up after each test method."""
        self.mock_embeddings_patcher.stop()
        
        # Clean up temporary files
        if os.path.exists(f"{self.index_path}.index"):
            os.remove(f"{self.index_path}.index")
        if os.path.exists(f"{self.index_path}.data"):
            os.remove(f"{self.index_path}.data")
    
    def test_initialization(self):
        """Test that the zettel note retriever initializes correctly."""
        self.assertEqual(self.zettel_retriever.note_class, CornellMethodNote)
        self.assertEqual(self.zettel_retriever.class_name, "CornellMethodNote")
        self.assertEqual(self.zettel_retriever.embedding_dim, self.test_embedding_dim)
        self.assertEqual(self.zettel_retriever.index_path, self.index_path)
    
    def test_add_document(self):
        """Test adding a Cornell note to the retriever."""
        doc_id = self.zettel_retriever.add_document(self.sample_zettel_note)
        
        # Check that the document was added correctly
        self.assertEqual(doc_id, self.sample_zettel_note.id)
        self.assertIn(doc_id, self.zettel_retriever.notes)
        self.assertIn(doc_id, self.zettel_retriever.note_ids)
        self.assertIn(doc_id, self.zettel_retriever.metadata_dict)
        
        # Check that the embedding was generated and added to the index
        expected_content = f"{self.sample_zettel_note.content}"
        self.mock_embeddings.embed_query.assert_called_with(expected_content)
        self.assertEqual(self.zettel_retriever.index.ntotal, 1)
