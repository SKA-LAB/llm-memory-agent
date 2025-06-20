from fastapi import FastAPI, HTTPException, Body, Query
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel
import os
import json
from pathlib import Path

from utils.note_processor import NoteProcessor
from utils.retrievers import CornellNoteRetriever, ZettelNoteRetriever
from utils.community_detector import ZettelCommunityDetector

app = FastAPI(
    title="LLM Memory Agent API",
    description="API for processing texts into Cornell and Zettel notes with vector search capabilities",
    version="1.0.0"
)

# Models for API requests and responses
class TextProcessRequest(BaseModel):
    text: str
    source_id: Optional[str] = None
    index_name: str = "default"

class BatchProcessRequest(BaseModel):
    texts: List[str]
    source_ids: Optional[List[str]] = None
    index_name: str = "default"

class NoteResponse(BaseModel):
    cornell_note_id: str
    cornell_note_title: str
    zettel_note_ids: List[str]
    zettel_note_titles: List[str]

class IndexInfo(BaseModel):
    name: str
    cornell_notes_count: int
    zettel_notes_count: int
    created_at: str
    last_updated: str

class CommunityDetectionRequest(BaseModel):
    index_name: str
    algorithm: str = "louvain"
    resolution: float = 1.0
    min_community_size: int = 3
    visualize: bool = False
    output_path: Optional[str] = None
    skip_existing: bool = True  # Add this parameter

class CommunityDetectionResponse(BaseModel):
    communities_count: int
    synthesis_notes: List[Dict[str, Any]]
    visualization_path: Optional[str] = None

# Global storage for note processors by index name
note_processors: Dict[str, NoteProcessor] = {}
index_metadata: Dict[str, Dict[str, Any]] = {}

# Directory for storing indices
INDICES_DIR = Path(os.path.expanduser("indices"))
INDICES_DIR.mkdir(parents=True, exist_ok=True)

def get_or_create_note_processor(index_name: str) -> NoteProcessor:
    """Get an existing note processor or create a new one for the given index name"""
    if index_name in note_processors:
        return note_processors[index_name]
    
    # Create directory for this index
    index_dir = INDICES_DIR / index_name
    index_dir.mkdir(exist_ok=True)
    
    # Create paths for Cornell and Zettel indices
    cornell_index_path = str(index_dir / "cornell_index")
    zettel_index_path = str(index_dir / "zettel_index")
    
    # Create retrievers with specific paths for this index
    cornell_retriever = CornellNoteRetriever(index_path=cornell_index_path)
    zettel_retriever = ZettelNoteRetriever(index_path=zettel_index_path)
    
    # Create note processor
    processor = NoteProcessor(cornell_retriever=cornell_retriever, zettel_retriever=zettel_retriever)
    note_processors[index_name] = processor
    
    # Initialize or update metadata
    if index_name not in index_metadata:
        from datetime import datetime
        index_metadata[index_name] = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "cornell_notes_count": 0,
            "zettel_notes_count": 0
        }
    
    return processor

@app.post("/process", response_model=NoteResponse)
async def process_text(request: TextProcessRequest):
    """Process a single text into Cornell and Zettel notes"""
    processor = get_or_create_note_processor(request.index_name)
    
    # Process the text
    cornell_note, zettel_notes = processor.process_text(request.text, request.source_id)
    
    # Update metadata
    from datetime import datetime
    index_metadata[request.index_name]["last_updated"] = datetime.now().isoformat()
    index_metadata[request.index_name]["cornell_notes_count"] += 1
    index_metadata[request.index_name]["zettel_notes_count"] += len(zettel_notes)
    
    # Return response
    return NoteResponse(
        cornell_note_id=cornell_note.id,
        cornell_note_title=cornell_note.title,
        zettel_note_ids=[note.id for note in zettel_notes],
        zettel_note_titles=[note.note_simple.title for note in zettel_notes]
    )

@app.post("/process_batch", response_model=List[NoteResponse])
async def process_batch(request: BatchProcessRequest):
    """Process a batch of texts into Cornell and Zettel notes"""
    processor = get_or_create_note_processor(request.index_name)
    
    # Process the batch
    results = processor.process_text_batch(request.texts, request.source_ids)
    
    # Update metadata
    from datetime import datetime
    index_metadata[request.index_name]["last_updated"] = datetime.now().isoformat()
    index_metadata[request.index_name]["cornell_notes_count"] += len(results)
    index_metadata[request.index_name]["zettel_notes_count"] += sum(len(zettel_notes) for _, zettel_notes in results)
    
    # Return responses
    responses = []
    for cornell_note, zettel_notes in results:
        responses.append(NoteResponse(
            cornell_note_id=cornell_note.id,
            cornell_note_title=cornell_note.title,
            zettel_note_ids=[note.id for note in zettel_notes],
            zettel_note_titles=[note.note_simple.title for note in zettel_notes]
        ))
    
    return responses

@app.get("/indices", response_model=List[IndexInfo])
async def get_indices():
    """Get information about all available indices"""
    result = []
    for name, meta in index_metadata.items():
        result.append(IndexInfo(
            name=name,
            cornell_notes_count=meta["cornell_notes_count"],
            zettel_notes_count=meta["zettel_notes_count"],
            created_at=meta["created_at"],
            last_updated=meta["last_updated"]
        ))
    return result

@app.get("/search/{index_name}")
async def search_notes(
    index_name: str,
    query: str = Query(..., description="Search query"),
    k: int = Query(5, description="Number of results to return"),
    note_type: str = Query("both", description="Type of notes to search: 'cornell', 'zettel', or 'both'")
):
    """Search for notes in a specific index"""
    if index_name not in note_processors:
        raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")
    
    processor = note_processors[index_name]
    results = {}
    
    if note_type in ["cornell", "both"] and processor.cornell_retriever:
        cornell_results = processor.cornell_retriever.search(query, k=k)
        results["cornell_notes"] = cornell_results
    
    if note_type in ["zettel", "both"] and processor.zettel_retriever:
        zettel_results = processor.zettel_retriever.search(query, k=k)
        results["zettel_notes"] = zettel_results
    
    return results

@app.get("/source/{index_name}/{source_id}")
async def get_notes_by_source(index_name: str, source_id: str):
    """Get all notes associated with a specific source document"""
    if index_name not in note_processors:
        raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")
    
    processor = note_processors[index_name]
    results = processor.get_notes_by_source(source_id)
    
    # Convert to a more API-friendly format
    formatted_results = []
    for cornell_note, zettel_notes in results:
        formatted_results.append({
            "cornell_note": {
                "id": cornell_note.id,
                "title": cornell_note.title,
                "content": cornell_note.content,
                "summary": cornell_note.summary
            },
            "zettel_notes": [
                {
                    "id": note.id,
                    "title": note.note_simple.title,
                    "keywords": note.note_simple.keywords,
                    "body": note.note_simple.body
                }
                for note in zettel_notes
            ]
        })
    
    return formatted_results


@app.post("/detect_communities", response_model=CommunityDetectionResponse)
async def detect_communities(request: CommunityDetectionRequest):
    """
    Detect communities in the Zettel note graph and generate synthesis notes for each community.
    
    Parameters:
    - index_name: Name of the index to use
    - algorithm: Community detection algorithm ('louvain', 'label_propagation', or 'girvan_newman')
    - resolution: Resolution parameter for Louvain algorithm (higher = smaller communities)
    - min_community_size: Minimum number of notes required to generate a synthesis
    - visualize: Whether to generate a visualization of the communities
    - output_path: Path to save the visualization (if None and visualize=True, a default path will be used)
    - skip_existing: If True, skip communities that already have synthesis notes
    """
    if request.index_name not in note_processors:
        raise HTTPException(status_code=404, detail=f"Index '{request.index_name}' not found")
    
    processor = note_processors[request.index_name]
    
    # Create a community detector using the Zettel retriever from the processor
    community_detector = ZettelCommunityDetector(processor.zettel_retriever)
    
    # Generate synthesis notes
    synthesis_notes = community_detector.generate_synthesis_notes(
        min_community_size=request.min_community_size,
        algorithm=request.algorithm,
        resolution=request.resolution,
        skip_existing=request.skip_existing  # Pass the new parameter
    )
    
    # Get communities for the response
    communities = community_detector.detect_communities(
        algorithm=request.algorithm,
        resolution=request.resolution
    )
    
    # Generate visualization if requested
    visualization_path = None
    if request.visualize:
        if request.output_path:
            visualization_path = request.output_path
        else:
            # Create a default path in the index directory
            vis_dir = INDICES_DIR / request.index_name / "visualizations"
            vis_dir.mkdir(exist_ok=True, parents=True)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            visualization_path = str(vis_dir / f"communities_{timestamp}.png")
        
        # Generate the visualization
        community_detector.visualize_graph(output_path=visualization_path)
    
    # Update metadata
    index_metadata[request.index_name]["last_updated"] = datetime.now().isoformat()
    index_metadata[request.index_name]["zettel_notes_count"] += len(synthesis_notes)
    
    # Format the synthesis notes for the response
    formatted_synthesis_notes = []
    for note in synthesis_notes:
        formatted_synthesis_notes.append({
            "id": note.id,
            "title": note.note_simple.title,
            "keywords": note.note_simple.keywords,
            "body": note.note_simple.body,
            "source": note.note_simple.source,
            "links": note.links,
            "type": note.type
        })
    
    return CommunityDetectionResponse(
        communities_count=len(communities),
        synthesis_notes=formatted_synthesis_notes,
        visualization_path=visualization_path
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Save all indices and metadata when shutting down"""
    for index_name, processor in note_processors.items():
        if processor.cornell_retriever:
            processor.cornell_retriever.save_index()
        if processor.zettel_retriever:
            processor.zettel_retriever.save_index()
    
    # Save metadata
    metadata_path = INDICES_DIR / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(index_metadata, f)

# Load existing metadata on startup
@app.on_event("startup")
async def startup_event():
    """Load existing metadata on startup"""
    metadata_path = INDICES_DIR / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            global index_metadata
            index_metadata = json.load(f)