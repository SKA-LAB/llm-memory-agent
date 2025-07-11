import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
from collections import defaultdict

from utils.community_detector import ZettelCommunityDetector
from utils.cornell_zettel_memory_system import ZettelNote
from utils.retrievers import ZettelNoteRetriever

class TestZettelCommunityDetector(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock ZettelNoteRetriever
        self.mock_retriever = Mock(spec=ZettelNoteRetriever)
        
        # Create sample notes for testing
        self.sample_notes = {}
        for i in range(1, 10):
            note_id = f"note-{i}"
            note = Mock(spec=ZettelNote)
            note.id = note_id
            note.title = f"Note {i}"
            note.type = "zettel"
            note.links = []
            note.metadata = {}
            self.sample_notes[note_id] = note
        
        # Set up links between notes to form communities
        # Community 1: notes 1, 2, 3
        self.sample_notes["note-1"].links = ["note-2", "note-3"]
        self.sample_notes["note-2"].links = ["note-1", "note-3"]
        self.sample_notes["note-3"].links = ["note-1", "note-2"]
        
        # Community 2: notes 4, 5, 6, 7
        self.sample_notes["note-4"].links = ["note-5", "note-6", "note-7"]
        self.sample_notes["note-5"].links = ["note-4", "note-6"]
        self.sample_notes["note-6"].links = ["note-4", "note-5", "note-7"]
        self.sample_notes["note-7"].links = ["note-4", "note-6"]
        
        # Isolated notes: 8, 9
        
        # Configure the mock retriever
        self.mock_retriever.note_ids = list(self.sample_notes.keys())
        self.mock_retriever.get_note_by_id = lambda note_id: self.sample_notes.get(note_id)
        
        # Create the community detector with the mock retriever
        self.detector = ZettelCommunityDetector(self.mock_retriever)
    
    def test_initialization(self):
        """Test that the community detector initializes correctly."""
        self.assertEqual(self.detector.zettel_retriever, self.mock_retriever)
    
    def test_build_graph(self):
        """Test building the graph from Zettel notes."""
        graph = self.detector.build_graph()
        
        # Check that the graph has the correct number of nodes and edges
        self.assertEqual(graph.number_of_nodes(), 9)
        
        # Check that the correct edges exist
        self.assertTrue(graph.has_edge("note-1", "note-2"))
        self.assertTrue(graph.has_edge("note-1", "note-3"))
        self.assertTrue(graph.has_edge("note-2", "note-3"))
        self.assertTrue(graph.has_edge("note-4", "note-5"))
        self.assertTrue(graph.has_edge("note-4", "note-6"))
        self.assertTrue(graph.has_edge("note-4", "note-7"))
        self.assertTrue(graph.has_edge("note-5", "note-6"))
        self.assertTrue(graph.has_edge("note-6", "note-7"))
        
        # Check that isolated nodes exist
        self.assertIn("note-8", graph.nodes())
        self.assertIn("note-9", graph.nodes())
        
        # Check node attributes
        self.assertEqual(graph.nodes["note-1"]["title"], "Note 1")
        self.assertEqual(graph.nodes["note-1"]["type"], "zettel")
    
    @patch('networkx.Graph')
    def test_build_graph_empty(self, mock_graph_class):
        """Test building a graph when there are no notes."""
        # Configure the mock graph
        mock_graph_instance = Mock()
        mock_graph_instance.number_of_nodes.return_value = 0
        mock_graph_class.return_value = mock_graph_instance
        
        # Configure the mock retriever to return no notes
        empty_retriever = Mock(spec=ZettelNoteRetriever)
        empty_retriever.note_ids = []
        
        detector = ZettelCommunityDetector(empty_retriever)
        graph = detector.build_graph()
        
        # The graph should be empty
        self.assertEqual(graph.number_of_nodes(), 0)
    
    @patch('community.best_partition')
    def test_detect_communities_louvain(self, mock_best_partition):
        """Test detecting communities using the Louvain algorithm."""
        # Mock the community detection result
        mock_best_partition.return_value = {
            "note-1": 0, "note-2": 0, "note-3": 0,
            "note-4": 1, "note-5": 1, "note-6": 1, "note-7": 1,
            "note-8": 2, "note-9": 3
        }
        
        communities = self.detector.detect_communities(algorithm='louvain')
        
        # Check that the communities are correctly detected
        self.assertEqual(len(communities), 4)
        self.assertListEqual(sorted(communities[0]), ["note-1", "note-2", "note-3"])
        self.assertListEqual(sorted(communities[1]), ["note-4", "note-5", "note-6", "note-7"])
        self.assertListEqual(communities[2], ["note-8"])
        self.assertListEqual(communities[3], ["note-9"])
    
    @patch('networkx.algorithms.community.label_propagation_communities')
    def test_detect_communities_label_propagation(self, mock_lp_communities):
        """Test detecting communities using the Label Propagation algorithm."""
        # Mock the community detection result
        mock_lp_communities.return_value = [
            {"note-1", "note-2", "note-3"},
            {"note-4", "note-5", "note-6", "note-7"},
            {"note-8", "note-9"}
        ]
        
        communities = self.detector.detect_communities(algorithm='label_propagation')
        
        # Check that the communities are correctly detected
        self.assertEqual(len(communities), 3)
        self.assertListEqual(sorted(communities[0]), ["note-1", "note-2", "note-3"])
        self.assertListEqual(sorted(communities[1]), ["note-4", "note-5", "note-6", "note-7"])
        self.assertListEqual(sorted(communities[2]), ["note-8", "note-9"])
    
    @patch('networkx.algorithms.community.girvan_newman')
    def test_detect_communities_girvan_newman(self, mock_gn):
        """Test detecting communities using the Girvan-Newman algorithm."""
        # Mock the community detection result
        mock_gn.return_value = [
            [{"note-1", "note-2", "note-3"}, {"note-4", "note-5", "note-6", "note-7"}, {"note-8"}, {"note-9"}]
        ]
        
        communities = self.detector.detect_communities(algorithm='girvan_newman')
        
        # Check that the communities are correctly detected
        self.assertEqual(len(communities), 4)
        self.assertListEqual(sorted(communities[0]), ["note-1", "note-2", "note-3"])
        self.assertListEqual(sorted(communities[1]), ["note-4", "note-5", "note-6", "note-7"])
        self.assertListEqual(communities[2], ["note-8"])
        self.assertListEqual(communities[3], ["note-9"])
    
    def test_detect_communities_unknown_algorithm(self):
        """Test detecting communities with an unknown algorithm."""
        communities = self.detector.detect_communities(algorithm='unknown')
        
        # Should return an empty dictionary
        self.assertEqual(communities, {})
    
    @patch('utils.community_detector.generate_synthesis_zettel')
    def test_generate_synthesis_notes(self, mock_generate_synthesis):
        """Test generating synthesis notes for communities."""
        # Mock the community detection
        self.detector.detect_communities = Mock(return_value={
            0: ["note-1", "note-2", "note-3"],
            1: ["note-4", "note-5", "note-6", "note-7"],
            2: ["note-8"],  # Too small, should be skipped
            3: ["note-9"]   # Too small, should be skipped
        })
        
        # Configure the mock retriever to return an empty list for get_notes_by_type
        self.mock_retriever.get_notes_by_type = Mock(return_value=[])
        
        # Mock the synthesis note generation
        mock_synthesis_notes = []
        for i in range(2):
            synthesis = Mock(spec=ZettelNote)
            synthesis.id = f"synthesis-{i}"
            synthesis.title = f"Synthesis {i}"
            synthesis.type = "synthesis"
            synthesis.links = []
            synthesis.metadata = {}
            mock_synthesis_notes.append(synthesis)
        
        mock_generate_synthesis.side_effect = mock_synthesis_notes
        
        # Generate synthesis notes
        synthesis_notes = self.detector.generate_synthesis_notes(min_community_size=3)
        
        # Check that synthesis notes were generated for the correct communities
        self.assertEqual(len(synthesis_notes), 2)
        mock_generate_synthesis.assert_called()
        self.assertEqual(mock_generate_synthesis.call_count, 2)
        
        # Check that the synthesis notes were added to the retriever
        self.assertEqual(self.mock_retriever.add_document.call_count, 2)
        
        # Check that metadata was added to the synthesis notes
        for note in synthesis_notes:
            self.assertIn('community_fingerprint', note.metadata)
            self.assertIn('algorithm', note.metadata)
            self.assertIn('resolution', note.metadata)
    
    @patch('utils.community_detector.generate_synthesis_zettel')
    def test_generate_synthesis_notes_skip_existing(self, mock_generate_synthesis):
        """Test skipping existing synthesis notes."""
        # Mock the community detection
        self.detector.detect_communities = Mock(return_value={
            0: ["note-1", "note-2", "note-3"],
            1: ["note-4", "note-5", "note-6", "note-7"]
        })
        
        # Create an existing synthesis note
        existing_synthesis = Mock(spec=ZettelNote)
        existing_synthesis.id = "existing-synthesis"
        existing_synthesis.title = "Existing Synthesis"
        existing_synthesis.type = "synthesis"
        existing_synthesis.links = []
        existing_synthesis.metadata = {'community_fingerprint': 'note-1,note-2,note-3'}
        
        # Mock the retriever to return the existing synthesis note
        self.mock_retriever.get_notes_by_type = Mock(return_value=[existing_synthesis])
        
        # Mock the synthesis note generation for the second community
        new_synthesis = Mock(spec=ZettelNote)
        new_synthesis.id = "new-synthesis"
        new_synthesis.title = "New Synthesis"
        new_synthesis.type = "synthesis"
        new_synthesis.links = []
        new_synthesis.metadata = {}
        
        mock_generate_synthesis.return_value = new_synthesis
        
        # Generate synthesis notes with skip_existing=True
        synthesis_notes = self.detector.generate_synthesis_notes(min_community_size=3, skip_existing=True)
        
        # Check that only one new synthesis note was generated
        self.assertEqual(len(synthesis_notes), 2)  # One existing, one new
        mock_generate_synthesis.assert_called_once()  # Only called for the second community
        
        # Check that the new synthesis note was added to the retriever
        self.mock_retriever.add_document.assert_called_once_with(new_synthesis)
    
    @patch('matplotlib.pyplot')
    def test_visualize_graph(self, mock_plt):
        """Test visualizing the graph."""
        # Mock the graph and community detection
        mock_graph = nx.Graph()
        self.detector.build_graph = Mock(return_value=mock_graph)
        self.detector.detect_communities = Mock(return_value={
            0: ["note-1", "note-2", "note-3"],
            1: ["note-4", "note-5", "note-6", "note-7"]
        })
        
        # Test visualization with output path
        self.detector.visualize_graph(output_path="test_output.png")
        # no need to assert anything, just making sure the function doesn't raise an exception

if __name__ == '__main__':
    unittest.main()