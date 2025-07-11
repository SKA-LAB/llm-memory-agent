import networkx as nx
from typing import List, Dict, Set, Optional
import logging
from collections import defaultdict
import matplotlib.pyplot as plt

from utils.retrievers import ZettelNoteRetriever
from utils.cornell_zettel_memory_system import ZettelNote, generate_synthesis_zettel

logger = logging.getLogger(__name__)

class ZettelCommunityDetector:
    """
    Detects communities in the graph created by bi-directional links between Zettel notes
    and generates synthesis notes for each community.
    """
    
    def __init__(self, zettel_retriever: ZettelNoteRetriever):
        """
        Initialize the community detector with a ZettelNoteRetriever.
        
        Args:
            zettel_retriever: Retriever for accessing Zettel notes
        """
        self.zettel_retriever = zettel_retriever
        
    def build_graph(self) -> nx.Graph:
        """
        Build a networkx graph from the Zettel notes and their links.
        
        Returns:
            A networkx Graph representing the Zettel notes and their connections
        """
        G = nx.Graph()
        
        # Get all note IDs from the retriever
        all_note_ids = self.zettel_retriever.note_ids
        
        # Add nodes for all notes
        for note_id in all_note_ids:
            note = self.zettel_retriever.get_note_by_id(note_id)
            if note:
                G.add_node(note_id, title=note.title, type=note.type)
        
        # Add edges for all links
        for note_id in all_note_ids:
            note = self.zettel_retriever.get_note_by_id(note_id)
            if note:
                for linked_id in note.links:
                    if linked_id in all_note_ids:  # Ensure the linked note exists
                        G.add_edge(note_id, linked_id)
        
        logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def detect_communities(self, algorithm: str = 'louvain', resolution: float = 1.0) -> Dict[int, List[str]]:
        """
        Detect communities in the Zettel note graph.
        
        Args:
            algorithm: Community detection algorithm to use ('louvain', 'label_propagation', or 'girvan_newman')
            resolution: Resolution parameter for the Louvain algorithm (higher values = smaller communities)
            
        Returns:
            Dictionary mapping community IDs to lists of note IDs
        """
        G = self.build_graph()
        
        if G.number_of_nodes() == 0:
            logger.warning("Graph is empty, no communities to detect")
            return {}
        
        communities = {}
        
        try:
            if algorithm == 'louvain':
                import community as community_louvain
                partition = community_louvain.best_partition(G, resolution=resolution)
                
                # Group nodes by community
                community_to_nodes = defaultdict(list)
                for node, community_id in partition.items():
                    community_to_nodes[community_id].append(node)
                
                communities = dict(community_to_nodes)
                
            elif algorithm == 'label_propagation':
                from networkx.algorithms import community
                result = community.label_propagation_communities(G)
                for i, comm in enumerate(result):
                    communities[i] = list(comm)
                    
            elif algorithm == 'girvan_newman':
                from networkx.algorithms import community
                # For Girvan-Newman, we'll take the first level of the hierarchy
                result = list(community.girvan_newman(G))
                if result:
                    # Take the first level of communities
                    first_level = result[0]
                    for i, comm in enumerate(first_level):
                        communities[i] = list(comm)
            else:
                logger.error(f"Unknown community detection algorithm: {algorithm}")
                return {}
                
        except ImportError as e:
            logger.error(f"Failed to import community detection library: {e}")
            logger.error("Make sure to install required packages: pip install python-louvain networkx")
            return {}
        except Exception as e:
            logger.error(f"Error during community detection: {e}")
            return {}
            
        logger.info(f"Detected {len(communities)} communities using {algorithm} algorithm")
        return communities
    
    def generate_synthesis_notes(self, min_community_size: int = 3, 
                                algorithm: str = 'louvain', 
                                resolution: float = 1.0,
                                skip_existing: bool = True) -> List[ZettelNote]:
        """
        Generate synthesis notes for each detected community.
        
        Args:
            min_community_size: Minimum number of notes required to generate a synthesis
            algorithm: Community detection algorithm to use
            resolution: Resolution parameter for community detection
            skip_existing: If True, skip communities that already have synthesis notes
        
        Returns:
            List of generated synthesis ZettelNotes
        """
        communities = self.detect_communities(algorithm, resolution)
        synthesis_notes = []
        
        # Track existing synthesis notes by their source community fingerprint
        existing_synthesis_by_community = {}
        if skip_existing:
            # Get all existing synthesis notes
            existing_synthesis_notes = self.zettel_retriever.get_notes_by_type("synthesis")
            
            # Map them by their source community
            for note in existing_synthesis_notes:
                # Extract community fingerprint from metadata if available
                if hasattr(note, 'metadata') and note.metadata and 'community_fingerprint' in note.metadata:
                    existing_synthesis_by_community[note.metadata['community_fingerprint']] = note
    
        for community_id, note_ids in communities.items():
            if len(note_ids) < min_community_size:
                logger.info(f"Skipping community {community_id} with only {len(note_ids)} notes")
                continue
            
            # Get the actual notes for this community
            community_notes = []
            for note_id in note_ids:
                note = self.zettel_retriever.get_note_by_id(note_id)
                if note and note.type != "synthesis":  # Skip existing synthesis notes
                    community_notes.append(note)
        
            if len(community_notes) < min_community_size:
                logger.info(f"Skipping community {community_id} with only {len(community_notes)} valid notes")
                continue
        
            # Create a fingerprint for this community (sorted note IDs)
            community_fingerprint = ",".join(sorted([note.id for note in community_notes]))
            
            # Check if we already have a synthesis for this exact community
            if skip_existing and community_fingerprint in existing_synthesis_by_community:
                existing_note = existing_synthesis_by_community[community_fingerprint]
                logger.info(f"Skipping already synthesized community with fingerprint {community_fingerprint}")
                synthesis_notes.append(existing_note)
                continue
            
            try:
                # Generate a synthesis note for this community
                synthesis_note = generate_synthesis_zettel(community_notes)
                
                # Add community fingerprint to metadata
                if not hasattr(synthesis_note, 'metadata'):
                    synthesis_note.metadata = {}
                synthesis_note.metadata['community_fingerprint'] = community_fingerprint
                synthesis_note.metadata['algorithm'] = algorithm
                synthesis_note.metadata['resolution'] = resolution
                
                # Add the synthesis note to the retriever
                self.zettel_retriever.add_document(synthesis_note)
                
                # Add the synthesis note to our results
                synthesis_notes.append(synthesis_note)
                
                logger.info(f"Generated synthesis note '{synthesis_note.title}' for community {community_id} with {len(community_notes)} notes")
            except Exception as e:
                logger.error(f"Failed to generate synthesis note for community {community_id}: {e}")
    
        return synthesis_notes
    
    def visualize_graph(self, output_path: Optional[str] = None) -> None:
        """
        Visualize the Zettel note graph with communities highlighted.
        
        Args:
            output_path: Path to save the visualization (if None, displays interactively)
        """
        try:
            G = self.build_graph()
            communities = self.detect_communities()
            
            # Create a mapping of nodes to community IDs
            node_community = {}
            for community_id, nodes in communities.items():
                for node in nodes:
                    node_community[node] = community_id
            
            # Set up the plot
            plt.figure(figsize=(12, 8))
            
            # Define positions for nodes using spring layout
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes colored by community
            for community_id in set(node_community.values()):
                nodes = [node for node in G.nodes() if node_community.get(node) == community_id]
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=100, 
                                      node_color=f"C{community_id % 10}", 
                                      label=f"Community {community_id}")
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, alpha=0.5)
            
            # Draw node labels
            node_labels = {node: G.nodes[node].get('title', node)[:10] + "..." 
                          for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
            
            plt.title("Zettel Note Communities")
            plt.legend()
            plt.axis('off')
            
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Graph visualization saved to {output_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.error("Visualization requires matplotlib. Install with: pip install matplotlib")
        except Exception as e:
            logger.error(f"Error during graph visualization: {e}")
