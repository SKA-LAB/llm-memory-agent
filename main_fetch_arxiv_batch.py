import datetime
from utils.arxiv_batch_fetcher import fetch_papers, get_fetched_papers

def main():
    start_date = datetime.date(2025, 3, 1)
    end_date = datetime.date.today()

    # Fetch papers
    fetch_papers(start_date, end_date)

    # Get the fetched papers
    get_fetched_papers()

if __name__ == "__main__":
    main()