import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logging(log_dir='logs'):
    """
    Set up logging configuration for the ML process.
    
    Args:
        log_dir (str): Directory to store log files
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger('ml_process')
    logger.setLevel(logging.DEBUG)
    
    # Create handlers
    # File handler for all logs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_log_file = Path(log_dir) / f'ml_process_{timestamp}.log'
    file_handler = logging.FileHandler(all_log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler for INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Add formatters to handlers
    file_handler.setFormatter(detailed_formatter)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create a default logger instance
logger = setup_logging() 