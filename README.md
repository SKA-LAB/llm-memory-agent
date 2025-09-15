# LLM Memory Agent

An intelligent memory management system that combines Cornell Method note-taking with Zettelkasten principles for organizing and retrieving knowledge from various sources.

## Overview

This system provides a comprehensive solution for capturing, organizing, and retrieving information using two complementary note-taking methodologies:

- **Cornell Method Notes**: Structured notes with cues, main content, and summaries
- **Zettel Notes**: Interconnected atomic notes following Zettelkasten principles

## Key Features

### 🧠 Dual Note-Taking System
- **Cornell Method Notes**: Perfect for structured learning and lecture notes
- **Zettel Notes**: Ideal for building interconnected knowledge graphs
- Seamless linking between both note types

### 🔍 Advanced Retrieval
- Vector-based semantic search using FAISS
- Embedding-powered similarity matching with Nomic embeddings
- Optional reranking for improved search quality
- Metadata-based filtering and organization

### 📄 Source Document Management
- Support for various document types and sources
- Automatic metadata extraction and management
- Source tracking and attribution for all notes

### 🔗 Knowledge Graph
- Bidirectional linking between notes
- Tag-based organization
- Relationship mapping between concepts

### 🚀 API Interface
- RESTful API for all operations
- Easy integration with external applications
- Programmatic access to all features

## Architecture

### System Overview

![LLM Memory Agent Architecture](figures/mermaid-diagram-llm-memory-agent.png)

The system follows a layered architecture with four distinct layers:

1. **Layer 1: Input & Ingestion** - Handles source documents through the Source Document Manager
2. **Layer 2: Processing & Memory** - Manages note structures and embeddings with the Memory System
3. **Layer 3: Knowledge Storage** - Stores structured notes and maintains the FAISS vector index
4. **Layer 4: Retrieval & Interface** - Provides search capabilities and user interfaces

### Core Components

1. **Memory System** (`utils/cornell_zettel_memory_system.py`)
   - Cornell Method Note and Zettel Note models
   - Core data structures and relationships

2. **Retrievers** (`utils/retrievers.py`)
   - FAISS-based vector retrieval
   - Specialized retrievers for each note type
   - Advanced search and filtering capabilities

3. **Source Document Manager** (`utils/source_document_manager.py`)
   - Document ingestion and management
   - Metadata extraction and storage

4. **API Layer** (`api/main.py`)
   - RESTful endpoints for all operations
   - Index management and statistics

### Note Types

#### Cornell Method Notes
```python
{
    "id": "unique_identifier",
    "note_simple": {
        "cues": ["key", "concepts"],
        "main_content": "detailed notes",
        "summary": "concise summary"
    },
    "content": "full text representation",
    "zettle_ids": ["linked_zettel_ids"],
    "source_id": "source_document_id",
    "source_title": "source_document_title"
}
```

#### Zettel Notes
```python
{
    "id": "unique_identifier",
    "note_simple": {
        "title": "note title",
        "keywords": ["relevant", "keywords"],
        "body": "main content"
    },
    "content": "full text representation",
    "links": ["connected_note_ids"],
    "tags": ["organizational", "tags"],
    "cornell_id": "linked_cornell_note_id"
}
```

## Getting Started

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd llm-memory-agent

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```python
from utils.cornell_zettel_memory_system import CornellMethodNote, ZettelNote
from utils.retrievers import CornellNoteRetriever, ZettelNoteRetriever

# Initialize retrievers
cornell_retriever = CornellNoteRetriever()
zettel_retriever = ZettelNoteRetriever()

# Create and add notes
cornell_note = CornellMethodNote(...)
cornell_retriever.add_document(cornell_note)

# Search for related content
results = cornell_retriever.search("machine learning concepts", k=5)
```

### API Usage
```python
python api/main.py
```
The API provides endpoints for:
- Creating and managing notes
- Searching and retrieval
- Source document management
- Index statistics and management

## Use Cases
- **Research Management:** Organize academic papers and research notes
- **Learning Systems:** Structure educational content with Cornell Method
- **Knowledge Bases:** Build interconnected knowledge graphs
- **Content Organization:** Manage and retrieve large collections of documents
- **Study Tools:** Create effective study materials with proven note-taking methods

## Testing
Run the test suite:
```bash
python -m pytest tests/
```
Tests cover:
- Note creation and management
- Retrieval functionality
- Source document handling
- API endpoints

## Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request