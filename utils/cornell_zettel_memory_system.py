from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime
import logging
from pydantic import BaseModel
from utils.init_llm import get_llm, parse_to_json

# Set up logger for this module
logger = logging.getLogger(__name__)


class CornellSimple(BaseModel):
    main_note: str
    questions: List[str]
    summary: str


class ZettelSimple(BaseModel):
    title: str
    keywords: List[str]
    body: str
    source: Optional[str] = None


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
    source_title: str = None  # Title of the source document that this cornell method note is generated from
    
    @property
    def title(self) -> str:
        """Get a title for the Cornell note based on the summary"""
        if hasattr(self.note_simple, 'summary'):
            # Extract first sentence or first 50 characters from summary
            summary = self.note_simple.summary
            title = summary.split('.')[0] if '.' in summary else summary[:50]
            logger.debug(f"Generated title for Cornell note {self.id[:8]}: {title}")
            return title + ("..." if len(title) >= 50 else "")
        logger.debug(f"Using default title for Cornell note {self.id[:8]}")
        return f"Cornell Note {self.id[:8]}"
    
    @property
    def summary(self) -> str:
        """Get the summary from the Cornell note"""
        if hasattr(self.note_simple, 'summary'):
            logger.debug(f"Retrieved summary for Cornell note {self.id[:8]}")
            return self.note_simple.summary
        logger.debug(f"No summary available for Cornell note {self.id[:8]}")
        return ""


class ZettelNote(BaseModel):
    """A Zettel note that represents a single unit of information in the memory system."""
    id: str
    created_at: datetime = datetime.now().isoformat()
    accessed_at: Optional[datetime] = None
    note_simple: ZettelSimple
    content: str  # string representation of the ZettelSimple object
    type: str = "standard"
    links: List[str] = []
    tags: List[str] = []
    retrieval_count: int = 0
    cornell_id: str = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def title(self) -> str:
        """Get the title from the Zettel note"""
        if hasattr(self.note_simple, 'title'):
            logger.debug(f"Retrieved title for Zettel note {self.id[:8]}")
            return self.note_simple.title
        logger.debug(f"Using default title for Zettel note {self.id[:8]}")
        return f"Zettel Note {self.id[:8]}"


def generate_cornell_method_note(text: str) -> CornellMethodNote:
    logger.info("Generating Cornell method note from text")
    logger.debug(f"Input text length: {len(text)} characters")
    
    prompt = get_cornell_method_prompt(text)
    llm = get_llm()
    
    logger.debug("Sending prompt to LLM for Cornell note generation")
    response = llm.invoke(prompt).content
    
    try:
        cornell_simple = parse_to_json(response, CornellSimple)
        logger.debug("Successfully parsed LLM response to CornellSimple")
        
        text_content = f"Main body:\n{cornell_simple.main_note}\n\nQuestions:\n{cornell_simple.questions}\n\nSummary:\n{cornell_simple.summary}"
        cornell_note = CornellMethodNote(
            note_simple=cornell_simple,
            content=text_content
        )
        logger.info(f"Created Cornell note with ID: {cornell_note.id}")
        return cornell_note
    except Exception as e:
        logger.error(f"Error parsing Cornell note from LLM response: {e}")
        raise


def get_cornell_method_prompt(text: str) -> str:
    logger.debug("Creating Cornell method prompt")
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
    logger.info(f"Generating Zettel notes from Cornell note {cornell_note.id}")
    
    cornell_text_content = cornell_note.content.strip()
    prompt = get_Zettel_prompt(cornell_text_content)
    llm = get_llm()
    
    logger.debug("Sending prompt to LLM for Zettel notes generation")
    response = llm.invoke(prompt).content
    
    try:
        Zettel_note_texts = parse_Zettel_response(response)
        logger.debug(f"Parsed {len(Zettel_note_texts)} Zettel notes from LLM response")
        
        Zettel_notes = []
        
        for i, note_text in enumerate(Zettel_note_texts):
            try:
                Zettel_note_simple = parse_to_json(note_text, ZettelSimple)
                note_text = f"Title: {Zettel_note_simple.title}\nKeywords: {Zettel_note_simple.keywords}\nBody: {Zettel_note_simple.body}\nSource: {Zettel_note_simple.source}"
                
                this_Zettel = ZettelNote(
                    id=uuid.uuid4().hex,  # Generate ID for Zettel note
                    note_simple=Zettel_note_simple,
                    content=note_text,
                    type="main",
                    cornell_id=cornell_note.id,
                )
                
                Zettel_notes.append(this_Zettel)
                cornell_note.zettle_ids.append(this_Zettel.id)
                logger.debug(f"Created Zettel note {i+1}/{len(Zettel_note_texts)} with ID: {this_Zettel.id}")
            except Exception as e:
                logger.error(f"Error parsing Zettel note {i+1}: {e}")
        
        logger.info(f"Successfully created {len(Zettel_notes)} Zettel notes from Cornell note {cornell_note.id}")
        return Zettel_notes
    except Exception as e:
        logger.error(f"Error generating Zettel notes: {e}")
        return []


def get_Zettel_prompt(cornell_note_content: str) -> str:
    logger.debug("Creating Zettel prompt")
    prompt = f"""
Take the following Cornell Note and break it down into multiple, atomic Zettelkasten notes:

---START CORNELL NOTE---
{cornell_note_content}
---END CORNELL NOTE---

Each Zettel should represent a single, distinct idea and have the following sections:

1.  **Title:** A concise, informative title that captures the essence of the information.
2.  **Keywords:** A list of important words or phrases that describe the content of the note.
3.  **Body:** A brief, self-contained paragpraph, often with a specific idea or concept.
4.  **Source:** The original source of the information.

Separate each generated Zettel note with '---ZETTEL_SEPARATOR---'. Do not include any other text before the first Zettel or after the last one.
"""
    return prompt


def parse_Zettel_response(response: str) -> List[str]:
    logger.debug("Parsing Zettel response from LLM")
    
    zettel_separator_index = response.find("---ZETTEL_SEPARATOR---")
    if zettel_separator_index == -1:
        logger.warning("No Zettel separator found in LLM response")
        return []
    
    zettel_content = response[zettel_separator_index + len("---ZETTEL_SEPARATOR---"):].strip()
    zettel_lines = zettel_content.split("\n")
    
    zettel_notes = []
    current_note = ""
    
    for line in zettel_lines:
        if line.strip() == "---ZETTEL_SEPARATOR---":
            if current_note:
                zettel_notes.append(current_note.strip())
                current_note = ""
        else:
            current_note += line + "\n"
    
    if current_note:
        zettel_notes.append(current_note.strip())
    
    logger.debug(f"Found {len(zettel_notes)} Zettel notes in LLM response")
    return zettel_notes


def get_synthesis_zettel_prompt(zettle_notes: List[ZettelNote]) -> str:
    logger.debug(f"Creating synthesis Zettel prompt for {len(zettle_notes)} notes")
    
    zettel_content = "\n------\n------\n".join([note.content for note in zettle_notes])
    prompt = f"""
Generate a synthesis Zettel for the following set of Zettel notes:

---START ZETTLES---
{zettel_content}
---END ZETTLES---

The synthesis Zettel identify the common theme in the set of notes above and should include the following sections:

1.  **Title:** A concise, informative title that captures the essence of the information.
2.  **Keywords:** A list of important words or phrases that describe the content of the note.
3.  **Body:** A brief, self-contained paragpraph, often with a specific idea or concept.
4.  **Source:** The original source of the information.
"""
    return prompt


def generate_synthesis_zettel(zettle_notes: List[ZettelNote]) -> ZettelNote:
    logger.info(f"Generating synthesis Zettel from {len(zettle_notes)} notes")
    
    if not zettle_notes:
        logger.warning("Cannot generate synthesis Zettel: no input notes provided")
        return None
    
    prompt = get_synthesis_zettel_prompt(zettle_notes)
    llm = get_llm()
    
    logger.debug("Sending prompt to LLM for synthesis Zettel generation")
    response = llm.invoke(prompt).content
    
    try:
        simple_note = parse_to_json(response.strip(), ZettelSimple)
        logger.debug("Successfully parsed LLM response to ZettelSimple")
        
        synthesis_note_text = f"Title: {simple_note.title}\nKeywords: {simple_note.keywords}\nBody: {simple_note.body}\nSource: {simple_note.source}"
        synthesis_note = ZettelNote(
            id=uuid.uuid4().hex,  # Generate ID for synthesis note
            note_simple=simple_note,
            content=synthesis_note_text,
            type="synthesis",
            links=[note.id for note in zettle_notes]
        )
        
        logger.info(f"Created synthesis Zettel note with ID: {synthesis_note.id}")
        return synthesis_note
    except Exception as e:
        logger.error(f"Error generating synthesis Zettel: {e}")
        raise