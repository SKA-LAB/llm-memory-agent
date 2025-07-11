import unittest
import os
import json
import tempfile
import shutil
from datetime import datetime
from unittest.mock import MagicMock, patch

from utils.source_document_manager import SourceDocumentManager, SourceDocument
from utils.retrievers import CornellNoteRetriever, ZettelNoteRetriever


class TestSourceDocumentManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        
        # Mock retrievers
        self.mock_cornell_retriever = MagicMock(spec=CornellNoteRetriever)
        self.mock_zettel_retriever = MagicMock(spec=ZettelNoteRetriever)
        
        # Create manager with test directory
        self.manager = SourceDocumentManager(
            cornell_retriever=self.mock_cornell_retriever,
            zettel_retriever=self.mock_zettel_retriever,
            storage_path=self.test_dir
        )
        
        # Sample document for testing
        self.sample_doc = SourceDocument(
            id="test123",
            title="Test Document",
            author="Test Author",
            content="This is test content",
            url="https://example.com/test",
            metadata={"category": "test"}
        )

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    def test_init_creates_storage_directory(self):
        """Test that initialization creates the storage directory if it doesn't exist."""
        new_path = os.path.join(self.test_dir, "new_dir")
        manager = SourceDocumentManager(storage_path=new_path)
        self.assertTrue(os.path.exists(new_path))

    def test_add_document(self):
        """Test adding a document."""
        doc_id = self.manager.add_document(self.sample_doc)
        
        # Check document was added to memory
        self.assertEqual(doc_id, "test123")
        self.assertIn(doc_id, self.manager.documents)
        
        # Check document was saved to file
        doc_file = os.path.join(self.test_dir, "source_documents.json")
        self.assertTrue(os.path.exists(doc_file))
        
        with open(doc_file, 'r') as f:
            saved_docs = json.load(f)
            self.assertEqual(len(saved_docs), 1)
            self.assertEqual(saved_docs[0]["id"], "test123")
            self.assertEqual(saved_docs[0]["title"], "Test Document")

    def test_get_document(self):
        """Test retrieving a document by ID."""
        self.manager.add_document(self.sample_doc)
        
        # Get existing document
        doc = self.manager.get_document("test123")
        self.assertIsNotNone(doc)
        self.assertEqual(doc.title, "Test Document")
        
        # Get non-existent document
        doc = self.manager.get_document("nonexistent")
        self.assertIsNone(doc)

    def test_delete_document(self):
        """Test deleting a document."""
        self.manager.add_document(self.sample_doc)
        
        # Delete existing document
        result = self.manager.delete_document("test123")
        self.assertTrue(result)
        self.assertNotIn("test123", self.manager.documents)
        
        # Delete non-existent document
        result = self.manager.delete_document("nonexistent")
        self.assertFalse(result)

    def test_get_related_notes(self):
        """Test getting notes related to a document."""
        # Setup mock Cornell notes
        mock_cornell_note1 = MagicMock()
        mock_cornell_note1.zettle_ids = ["zettel1", "zettel2"]
        
        mock_cornell_note2 = MagicMock()
        mock_cornell_note2.zettle_ids = ["zettel3"]
        
        # Setup mock Zettel notes
        mock_zettel1 = MagicMock()
        mock_zettel2 = MagicMock()
        mock_zettel3 = MagicMock()
        
        # Configure retrievers
        self.mock_cornell_retriever.get_notes_by_source_id.return_value = [
            mock_cornell_note1, mock_cornell_note2
        ]
        
        self.mock_zettel_retriever.get_note_by_id.side_effect = lambda id: {
            "zettel1": mock_zettel1,
            "zettel2": mock_zettel2,
            "zettel3": mock_zettel3
        }.get(id)
        
        # Get related notes
        result = self.manager.get_related_notes("test123")
        
        # Check results
        self.assertEqual(len(result["cornell_notes"]), 2)
        self.assertEqual(len(result["zettel_notes"]), 3)
        
        # Verify the retrievers were called correctly
        self.mock_cornell_retriever.get_notes_by_source_id.assert_called_once_with("test123")
        self.assertEqual(self.mock_zettel_retriever.get_note_by_id.call_count, 3)

    def test_create_document_from_text(self):
        """Test creating a document from text."""
        doc = self.manager.create_document_from_text(
            title="New Document",
            content="New content",
            author="New Author",
            url="https://example.com/new",
            metadata={"category": "new"}
        )
        
        # Check document properties
        self.assertEqual(doc.title, "New Document")
        self.assertEqual(doc.content, "New content")
        self.assertEqual(doc.author, "New Author")
        self.assertEqual(doc.url, "https://example.com/new")
        self.assertEqual(doc.metadata, {"category": "new"})
        
        # Check document was added to manager
        self.assertIn(doc.id, self.manager.documents)

    def test_load_documents(self):
        """Test loading documents from storage."""
        # Create a test document file
        doc_file = os.path.join(self.test_dir, "source_documents.json")
        test_docs = [
            {
                "id": "doc1",
                "title": "Document 1",
                "author": "Author 1",
                "content": "Content 1",
                "metadata": {},
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "doc2",
                "title": "Document 2",
                "author": "Author 2",
                "content": "Content 2",
                "metadata": {"key": "value"},
                "created_at": datetime.now().isoformat()
            }
        ]
        
        with open(doc_file, 'w') as f:
            json.dump(test_docs, f)
        
        # Create a new manager to load the documents
        new_manager = SourceDocumentManager(storage_path=self.test_dir)
        
        # Check documents were loaded
        self.assertEqual(len(new_manager.documents), 2)
        self.assertIn("doc1", new_manager.documents)
        self.assertIn("doc2", new_manager.documents)
        self.assertEqual(new_manager.documents["doc1"].title, "Document 1")
        self.assertEqual(new_manager.documents["doc2"].title, "Document 2")

    @patch('utils.source_document_manager.logger')
    def test_load_documents_error_handling(self, mock_logger):
        """Test error handling when loading documents."""
        # Create an invalid JSON file
        doc_file = os.path.join(self.test_dir, "source_documents.json")
        with open(doc_file, 'w') as f:
            f.write("This is not valid JSON")
        
        # Create a new manager to attempt loading the invalid file
        new_manager = SourceDocumentManager(storage_path=self.test_dir)
        
        # Check error was logged
        mock_logger.error.assert_called_once()
        self.assertIn("Error loading source documents", mock_logger.error.call_args[0][0])
        
        # Check no documents were loaded
        self.assertEqual(len(new_manager.documents), 0)



if __name__ == '__main__':
    unittest.main()