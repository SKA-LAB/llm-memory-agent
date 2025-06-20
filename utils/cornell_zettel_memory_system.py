from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime
from pydantic import BaseModel
from init_llm import get_llm, parse_to_json


class CornellSimple(BaseModel):
    main_note: str
    questions: List[str]
    summary: str


class ZettelSimple(BaseModel):
    title: str
    keywords: List[str]
    body: str
    source: str


class CornellMethodNote(BaseModel):
    """A Cornell Method note that represents a single unit of information in the memory system."""
    id: str = uuid.uuid4().hex
    created_at: datetime = datetime.now().isoformat()
    accessed_at: Optional[datetime] = None
    retrieval_count: int = 0
    note_simple: CornellSimple
    content: str  # string representation of the CornellSimple object
    zettle_ids: List[str] = []
    source_id: str = None  # ID of the source document that this cornell method note is generated from


class ZettelNote(BaseModel):
    """A Zettel note that represents a single unit of information in the memory system."""
    id: str = uuid.uuid4().hex
    created_at: datetime = datetime.now().isoformat()
    accessed_at: Optional[datetime] = None
    note_simple: ZettelSimple
    content: str  # string representation of the ZettelSimple object
    type: str = "main"  # or "synthesis"
    links: List[str] = []
    tags: List[str] = []
    retrieval_count: int = 0
    cornell_id: str = None


def generate_cornell_method_note(text: str) -> CornellMethodNote:
    prompt = get_cornell_method_prompt(text)
    llm = get_llm()
    response = llm.invoke(prompt).content
    cornell_simple = parse_to_json(response, CornellSimple)
    text_content = f"Main body:\n{cornell_simple.main_note}\n\nQuestions:\n{cornell_simple.questions}\n\nSummary:\n{cornell_simple.summary}"
    cornell_note = CornellMethodNote(
        note_simple = cornell_simple,
        content = text_content
    )
    return cornell_note

def get_cornell_method_prompt(text: str) -> str:
    prompt = f"""
Analyze the following text and structure it into a comprehensive Cornell Method note.
Your output must be in a clean, easily parsable format with three distinct, clearly labeled sections:

1.  **## Main Notes:** A detailed, point-by-point breakdown of the key information and concepts.
2.  **## Questions:** A list of high-level questions that are answered by the main notes.
3.  **## Summary:** A concise 1-3 sentence summary of the entire text's core message.

Here is the text to process:
{text}
"""
    return prompt

def get_Zettel_notes(cornell_note: CornellMethodNote) -> List[ZettelNote]:
    cornell_text_content = cornell_note.content.strip()
    prompt = get_Zettel_prompt(cornell_text_content)
    llm = get_llm()
    response = llm.invoke(prompt).content
    Zettel_note_texts = parse_Zettel_response(response)
    Zettel_notes_simple = [parse_to_json(note_text, ZettelSimple) for note_text in Zettel_note_texts]
    Zettel_notes = []
    for note in Zettel_notes_simple:
        note_text = f"Title: {note.title}\nKeywords: {note.keywords}\nBody: {note.body}\nSource: {note.source}"
        this_Zettel = ZettelNote(
            note_simple = note,
            content = note_text,
            type = "main",
            cornell_id = cornell_note.id,
        )
        Zettel_notes.append(this_Zettel)
        cornell_note.zettle_ids.append(this_Zettel.id)
    return Zettel_notes

def get_Zettel_prompt(cornell_note_content: str) -> str:
    prompt = f"""
Take the following Cornell Note and break it down into multiple, atomic Zettelkasten notes:

---START CORNELL NOTE---
{cornell_note_content}
---END CORNELL NOTE---

Each Zettel should represent a single, distinct idea and have the following sections:

1.  **Title:** A concise, informative title that captures the essence of the information.
2.  **Keywords:** A list of important words or phrases that describe the content of the note.
3.  **Body:** A brief, self-contained paragprah, often with a specific idea or concept.
4.  **Source:** The original source of the information.

Separate each generated Zettel note with '---ZETTEL_SEPARATOR---'. Do not include any other text before the first Zettel or after the last one.
"""
    return prompt


def parse_Zettel_response(response: str) -> List[str]:
    zettel_separator_index = response.find("---ZETTEL_SEPARATOR---")
    if zettel_separator_index == -1:
        return []
    zettel_content = response[zettel_separator_index + len("---ZETTEL_SEPARATOR---"):].strip()
    zettel_lines = zettel_content.split("\n")
    zettel_notes = []
    current_note = ""
    for line in zettel_lines:
        if line.strip() == "":
            if current_note:
                zettel_notes.append(current_note)
                current_note = ""
        else:
            current_note += line + "\n"
    if current_note:
        zettel_notes.append(current_note)