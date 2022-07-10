"""
This is the folder where all code about revieving the right model, dataset or network architecture is stored

"""

from .model_handler import get_model
from .database_handler import get_db
from .adaptation_method_handler import get_adapt_method

__all__ = ["get_model", "get_db", "get_adapt_method"]
