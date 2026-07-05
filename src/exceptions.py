class DataValidationError(Exception):
    """Raised when dataset validation fails."""
    pass


class ModelTrainingError(Exception):
    """Raised when model training fails."""
    pass


class PredictionError(Exception):
    """Raised when prediction fails."""
    pass