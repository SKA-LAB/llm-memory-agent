import arxiv
import datetime
import os
import time
import pickle
import fitz
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Get a logger for this file
logger = logging.getLogger(__name__)

# Configurations
CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.MA", "cs.NE", "cs.CV"]  # AI, Machine Learning, Computational Linguistics, Multiagent Systems, Neural and Evolutionary Computing
KEYWORDS = ["artificial intelligence", "large language model", "agent system", "reinforcement learning", "LLM", "AI agent", "deep learning"]
MAX_RESULTS_PER_QUERY = 100
SAVE_PROGRESS_FILE = "arxiv_fetch_progress.pkl"
OUTPUT_FILE = "arxiv_papers.pkl"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    return text

def initialize_arxiv_client():
    """Initialize and return an ArXiv client."""
    return arxiv.Client()

def load_progress():
    """Load progress from a file if it exists, otherwise return an empty set."""
    if os.path.exists(SAVE_PROGRESS_FILE):
        with open(SAVE_PROGRESS_FILE, "rb") as f:
            return pickle.load(f)
    return set()

def save_progress(processed_dates):
    """Save the progress to a file."""
    with open(SAVE_PROGRESS_FILE, "wb") as f:
        pickle.dump(processed_dates, f)

def load_papers():
    """Load previously fetched papers if they exist, otherwise return an empty list."""
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "rb") as f:
            return pickle.load(f)
    return []

def save_papers(papers):
    """Save the fetched papers to a file."""
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(papers, f)

def load_fetched_ids():
    """Load the set of already fetched paper IDs."""
    fetched_ids_file = "fetched_paper_ids.pkl"
    if os.path.exists(fetched_ids_file):
        with open(fetched_ids_file, "rb") as f:
            return pickle.load(f)
    return set()

def save_fetched_ids(fetched_ids):
    """Save the set of fetched paper IDs."""
    fetched_ids_file = "fetched_paper_ids.pkl"
    with open(fetched_ids_file, "wb") as f:
        pickle.dump(fetched_ids, f)

def fetch_batch(client, start_date, end_date, category, fetched_ids):
    """Fetch a batch of ArXiv papers within a date range for a specific category."""
    keyword_query = " OR ".join(f'"{kw}"' for kw in KEYWORDS)
    search_query = f"cat:{category} AND ({keyword_query}) AND submittedDate:[{start_date} TO {end_date}]"
    search = arxiv.Search(query=search_query, max_results=MAX_RESULTS_PER_QUERY, sort_by=arxiv.SortCriterion.SubmittedDate)

    papers = []
    for result in client.results(search):
        if result.entry_id in fetched_ids:
            continue  # Skip already fetched papers
        paper_data = {
            "id": result.entry_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary,
            "categories": result.primary_category,
            "pdf_url": result.pdf_url
        }

        # Download and extract PDF text
        pdf_path = f"./downloads/{result.entry_id.split('/')[-1]}.pdf"
        result.download_pdf(filename=pdf_path)
        paper_data["full_text"] = extract_text_from_pdf(pdf_path)
        papers.append(paper_data)
        fetched_ids.add(result.entry_id)  # Add the ID to the set of fetched papers
        os.remove(pdf_path)  # Remove downloaded PDF file to save disk space

    return papers

def fetch_papers(start_date, end_date):
    """Fetch papers for the given date range."""
    client = initialize_arxiv_client()
    processed_dates = load_progress()
    all_papers = load_papers()
    fetched_ids = load_fetched_ids()

    current_date = start_date
    while current_date < end_date:
        next_date = (current_date + datetime.timedelta(days=1)).strftime("%Y%m%d") + "0000"
        current_str = current_date.strftime("%Y%m%d") + "0000"

        if current_str in processed_dates:
            logger.info(f"Skipping already processed date: {current_str}")
            current_date += datetime.timedelta(days=1)
            continue

        logger.info(f"Fetching papers from {current_str} to {next_date}...")

        for category in CATEGORIES:
            try:
                papers = fetch_batch(client, current_str, next_date, category, fetched_ids)
                all_papers.extend(papers)
                logger.info(f"Fetched {len(papers)} papers for category {category}.")
            except Exception as e:
                logger.info(f"Error fetching {category}: {e}")

            time.sleep(5)  # Avoid hitting API rate limits

        # Save progress
        processed_dates.add(current_str)
        save_progress(processed_dates)
        save_papers(all_papers)
        save_fetched_ids(fetched_ids)

        current_date += datetime.timedelta(days=1)

    print(f"Total unique papers fetched: {len(all_papers)}")

def get_fetched_papers():
    """Return the list of fetched papers."""
    return load_papers()
