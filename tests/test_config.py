import logging

def configure_test_logging():
    """Configure logging for tests to show debug messages."""
    # Configure the root logger to show debug messages
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # You can also configure specific loggers if needed
    # For example, to focus on just the retrievers module:
    retriever_logger = logging.getLogger('utils.retrievers')
    retriever_logger.setLevel(logging.DEBUG)
import logging

def configure_test_logging():
    """Configure logging for tests to show debug messages."""
    # Configure the root logger to show debug messages
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # You can also configure specific loggers if needed
    # For example, to focus on just the retrievers module:
    retriever_logger = logging.getLogger('utils.retrievers')
    retriever_logger.setLevel(logging.DEBUG)
