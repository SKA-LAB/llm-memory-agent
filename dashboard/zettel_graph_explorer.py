import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import tempfile
import time

# Add the parent directory to the path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

from utils.retrievers import ZettelNoteRetriever, CornellNoteRetriever
from utils.note_processor import NoteProcessor
from utils.source_document_manager import SourceDocumentManager
from utils.community_detector import ZettelCommunityDetector

# Import MarkItDown for document conversion
try:
    from markitdown import MarkItDown
except ImportError:
    st.error("MarkItDown library is not installed. Please install it using: pip install markitdown")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Zettel Graph Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
INDICES_DIR = Path(os.environ.get("INDICES_DIR", Path.home() / "zettel_indices"))

def load_indices() -> List[str]:
    """Load available indices from the indices directory"""
    if not INDICES_DIR.exists():
        return []
    
    return [d.name for d in INDICES_DIR.iterdir() if d.is_dir()]

def get_note_processor(index_name: str) -> Optional[NoteProcessor]:
    """Get a note processor for the specified index"""
    if not index_name:
        return None
    
    index_dir = INDICES_DIR / index_name
    if not index_dir.exists():
        return None
    
    cornell_index_path = str(index_dir / "cornell_index")
    zettel_index_path = str(index_dir / "zettel_index")
    
    cornell_retriever = CornellNoteRetriever(index_path=cornell_index_path)
    zettel_retriever = ZettelNoteRetriever(index_path=zettel_index_path)
    
    return NoteProcessor(cornell_retriever=cornell_retriever, zettel_retriever=zettel_retriever)

def build_graph(zettel_retriever: ZettelNoteRetriever) -> Tuple[nx.Graph, Dict]:
    """Build a networkx graph from the Zettel notes and their links"""
    G = nx.Graph()
    node_data = {}
    
    # Get all note IDs from the retriever
    all_note_ids = zettel_retriever.note_ids
    
    # Add nodes for all notes
    for note_id in all_note_ids:
        note = zettel_retriever.get_note_by_id(note_id)
        if note:
            G.add_node(note_id)
            node_data[note_id] = {
                'title': note.title,
                'type': note.type,
                'content': note.content,
                'keywords': note.keywords if hasattr(note, 'keywords') else [],
                'cornell_id': note.cornell_id if hasattr(note, 'cornell_id') else None
            }
    
    # Add edges for all links
    for note_id in all_note_ids:
        note = zettel_retriever.get_note_by_id(note_id)
        if note:
            for linked_id in note.links:
                if linked_id in all_note_ids:  # Ensure the linked note exists
                    G.add_edge(note_id, linked_id)
    
    return G, node_data

def visualize_graph(G: nx.Graph, node_data: Dict, selected_node: Optional[str] = None) -> plt.Figure:
    """Create a visualization of the Zettel note graph"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define positions for nodes using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Define node colors based on type
    node_colors = []
    for node in G.nodes():
        if node == selected_node:
            node_colors.append('yellow')  # Highlight selected node
        elif node_data[node]['type'] == 'synthesis':
            node_colors.append('red')
        else:
            node_colors.append('skyblue')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # Draw node labels
    node_labels = {node: node_data[node]['title'][:15] + "..." if len(node_data[node]['title']) > 15 
                  else node_data[node]['title'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    
    # Add legend
    standard_patch = mpatches.Patch(color='skyblue', label='Standard Zettel')
    synthesis_patch = mpatches.Patch(color='red', label='Synthesis Zettel')
    selected_patch = mpatches.Patch(color='yellow', label='Selected Zettel')
    plt.legend(handles=[standard_patch, synthesis_patch, selected_patch], loc='upper right')
    
    plt.title("Zettel Note Graph")
    plt.axis('off')
    
    return fig

def display_zettel_details(zettel_note, cornell_retriever, source_manager=None):
    """Display details of a selected Zettel note"""
    st.subheader("Zettel Note Details")
    
    st.markdown(f"**Title:** {zettel_note.title}")
    st.markdown(f"**Type:** {zettel_note.type}")
    
    if hasattr(zettel_note, 'keywords') and zettel_note.keywords:
        st.markdown(f"**Keywords:** {', '.join(zettel_note.keywords)}")
    
    st.markdown("**Content:**")
    st.markdown(zettel_note.content)
    
    st.markdown(f"**Links:** {', '.join(zettel_note.links) if zettel_note.links else 'None'}")
    
    # Display metadata if available
    if hasattr(zettel_note, 'metadata') and zettel_note.metadata:
        st.markdown("**Metadata:**")
        st.json(zettel_note.metadata)
    
    # Display parent Cornell note if available
    if hasattr(zettel_note, 'cornell_id') and zettel_note.cornell_id:
        st.markdown("---")
        st.subheader("Parent Cornell Note")
        
        cornell_note = cornell_retriever.get_note_by_id(zettel_note.cornell_id)
        if cornell_note:
            st.markdown(f"**Title:** {cornell_note.title}")
            
            if hasattr(cornell_note, 'cues'):
                st.markdown("**Cues:**")
                st.markdown(cornell_note.cues)
            
            if hasattr(cornell_note, 'notes'):
                st.markdown("**Notes:**")
                st.markdown(cornell_note.notes)
            
            if hasattr(cornell_note, 'summary'):
                st.markdown("**Summary:**")
                st.markdown(cornell_note.summary)
            
            # Display source document if available
            if hasattr(cornell_note, 'source_id') and cornell_note.source_id and source_manager:
                st.markdown("---")
                st.subheader("Source Document")
                
                source_doc = source_manager.get_document_by_id(cornell_note.source_id)
                if source_doc:
                    st.markdown(f"**Title:** {source_doc.title}")
                    st.markdown(f"**Authors:** {', '.join(source_doc.authors)}")
                    st.markdown(f"**Published:** {source_doc.published}")
                    
                    if hasattr(source_doc, 'abstract'):
                        st.markdown("**Abstract:**")
                        st.markdown(source_doc.abstract)
                    
                    if hasattr(source_doc, 'url') and source_doc.url:
                        st.markdown(f"[View Original Document]({source_doc.url})")
                else:
                    st.warning("Source document not found")
        else:
            st.warning("Parent Cornell note not found")

def convert_to_markdown(file_path):
    """Convert a file to markdown using MarkItDown"""
    try:
        # Use MarkItDown to convert the file to markdown
        if not os.path.basename(file_path).endswith(('.txt', '.md')):
            md = MarkItDown(enable_plugins=False)
            result = md.convert(file_path)
            markdown_content = result.text_content
            return markdown_content
        else:
            # If the file is already markdown, simply read it as utf8
            with open(file_path, 'r', encoding='utf8') as f:
                markdown_content = f.read()
                return markdown_content
    except Exception as e:
        st.error(f"Error converting file {file_path}: {str(e)}")
        return None

def chunk_text(text, chunk_size=5000, overlap=500):
    """Split text into chunks with overlap"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # If we're not at the beginning, include overlap
        if start > 0:
            start = max(0, start - overlap)
        
        # Extract the chunk
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move to the next chunk
        start = end
    
    return chunks

def process_files(files, index_name):
    """Process uploaded files and add them to the index"""
    # Create or get the note processor for this index
    processor = get_note_processor(index_name)
    if not processor:
        print(f"Note processor for index {index_name} does not exist. Creating...")
        # Create the index directory
        index_dir = INDICES_DIR / index_name
        index_dir.mkdir(exist_ok=True, parents=True)
        
        # Create paths for Cornell and Zettel indices
        cornell_index_path = str(index_dir / "cornell_index")
        zettel_index_path = str(index_dir / "zettel_index")
        
        # Create retrievers with specific paths for this index
        cornell_retriever = CornellNoteRetriever(index_path=cornell_index_path)
        zettel_retriever = ZettelNoteRetriever(index_path=zettel_index_path)
        
        # Create note processor
        processor = NoteProcessor(cornell_retriever=cornell_retriever, zettel_retriever=zettel_retriever)
    else:
        print(f"Note processor for index {index_name} already exists")
    
    # Process each file
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(files):
        status_text.text(f"Processing file {i+1}/{len(files)}: {file.name}")
        
        # Save the file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name
        
        try:
            # Convert to markdown
            markdown_content = convert_to_markdown(tmp_path)
            if markdown_content:
                # Chunk the content
                chunks = chunk_text(markdown_content)
                
                # Process each chunk
                for j, chunk in enumerate(chunks):
                    chunk_title = f"{file.name} - Chunk {j+1}"
                    status_text.text(f"Processing file {i+1}/{len(files)}: {file.name} - Chunk {j+1}/{len(chunks)}")
                    
                    # Process the chunk to create Cornell and Zettel notes
                    processor.process_text(chunk, source_title=chunk_title, source_id=file.name)
            
            # Clean up the temporary file
            os.unlink(tmp_path)
        except Exception as e:
            st.error(f"Error processing file {file.name}: {str(e)}")
        
        # Update progress
        progress_bar.progress((i + 1) / len(files))
    
    status_text.text("All files processed. Running community detection...")
    
    # Run community detection
    try:
        community_detector = ZettelCommunityDetector(processor.zettel_retriever)
        synthesis_notes = community_detector.generate_synthesis_notes(
            min_community_size=3,
            algorithm='louvain',
            resolution=1.0
        )
        status_text.text(f"Community detection complete. Generated {len(synthesis_notes)} synthesis notes.")
    except Exception as e:
        st.error(f"Error during community detection: {str(e)}")
        status_text.text("Community detection failed.")
    
    return processor

def explorer_tab():
    """Content for the Explorer tab"""
    st.title("Zettel Graph Explorer")
    
    # Sidebar for index selection
    st.sidebar.header("Settings")
    
    indices = load_indices()
    if not indices:
        st.error("No indices found. Please create an index first.")
        return
    
    selected_index = st.sidebar.selectbox("Select Index", indices)
    
    # Get note processor for the selected index
    processor = get_note_processor(selected_index)
    if not processor or not processor.zettel_retriever:
        st.error(f"Failed to load index '{selected_index}'")
        return
    
    # Build the graph
    G, node_data = build_graph(processor.zettel_retriever)
    
    # Display graph statistics
    st.sidebar.subheader("Graph Statistics")
    st.sidebar.markdown(f"**Nodes:** {G.number_of_nodes()}")
    st.sidebar.markdown(f"**Edges:** {G.number_of_edges()}")
    
    # Filter options
    st.sidebar.subheader("Filters")
    show_synthesis = st.sidebar.checkbox("Show Synthesis Notes", value=True)
    show_standard = st.sidebar.checkbox("Show Standard Notes", value=True)
    
    # Apply filters
    filtered_nodes = []
    for node in G.nodes():
        node_type = node_data[node]['type']
        if (node_type == 'synthesis' and show_synthesis) or (node_type != 'synthesis' and show_standard):
            filtered_nodes.append(node)
    
    filtered_G = G.subgraph(filtered_nodes)
    
    # Main content area with two columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Zettel Note Graph")
        
        # Create visualization
        fig = visualize_graph(filtered_G, node_data, selected_node=st.session_state.get('selected_node'))
        st.pyplot(fig)
        
        # Node selection via dropdown
        node_options = [(node, f"{node_data[node]['title']} ({node_data[node]['type']})") 
                        for node in filtered_nodes]
        node_dict = {name: id for id, name in node_options}
        
        selected_node_name = st.selectbox(
            "Select a node to view details",
            options=[name for _, name in node_options],
            index=0 if node_options else None
        )
        
        if selected_node_name:
            selected_node_id = node_dict[selected_node_name]
            st.session_state['selected_node'] = selected_node_id
    
    with col2:
        if 'selected_node' in st.session_state:
            selected_node = st.session_state['selected_node']
            zettel_note = processor.zettel_retriever.get_note_by_id(selected_node)
            
            if zettel_note:
                # Try to create a source document manager if possible
                source_manager = None
                try:
                    source_manager = SourceDocumentManager()
                except:
                    pass
                
                display_zettel_details(
                    zettel_note, 
                    processor.cornell_retriever, 
                    source_manager
                )
            else:
                st.warning("Selected note not found")

def uploader_tab():
    """Content for the Uploader tab"""
    st.title("Upload Files to Index")
    
    # Sidebar for index selection
    st.sidebar.header("Settings")
    
    indices = load_indices()
    index_name = st.sidebar.text_input("Enter or create an index name", value="default_index" if not indices else indices[0])
    
    # File uploader
    uploaded_files = st.file_uploader("Upload files (.pdf, .docx, .txt, .md)", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    
    if st.button("Process Files"):
        if not uploaded_files:
            st.warning("Please upload files before processing.")
        else:
            if len(uploaded_files) > 200:
                st.error("You can upload up to 200 files at once.")
            else:
                process_files(uploaded_files, index_name)
                st.success("Files processed and indexed successfully.")

def main():
    # Create tabs
    tab1, tab2 = st.tabs(["Explorer", "Uploader"])
    
    with tab1:
        explorer_tab()
    
    with tab2:
        uploader_tab()

if __name__ == "__main__":
    main()