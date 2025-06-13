"""
Logging Configuration Module
Sets up structured logging for the anime shorts generator
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

def setup_logger(name: str = "animegen", level: str = "INFO", 
                log_file: Optional[str] = None, verbose: bool = False) -> logging.Logger:
    """
    Set up a logger with console and optional file output
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        verbose: If True, use DEBUG level and detailed format
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set level
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Create formatters
    if verbose:
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
        )
    else:
        console_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(name)

class LoggerMixin:
    """Mixin class to add logging capability to any class"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        return self._logger