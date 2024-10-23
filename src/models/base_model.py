import yaml

from pathlib import Path
from abc import ABC, abstractmethod

from utils import load_config

class BaseModel(ABC):
    def __init__(self, config_path=Path.cwd() / 'config' / 'model_config.yaml'):
        self._load_config(config_path)
       
    def _load_config(self, config_path):
        if config_path:
            self.config = load_config(config_path)['models']
        else:
            self.config = {}

    @abstractmethod
    def train(self, train_loader):
        """Train the model using the given training data."""
        pass

    @abstractmethod
    def evaluate(self, test_loader):
        """Evaluate the model's performance using the given test data."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions on new data."""
        pass

    @abstractmethod
    def hyperparameter_tuning(self, train_loader, test_loader, param_grid):
        """Perform hyperparameter tuning."""
        pass

    @abstractmethod
    def save_model(self, file_path):
        """Save the model to the specified file."""
        pass

    @abstractmethod
    def load_model(self, file_path):
        """Load a model from the specified file."""
        pass

    @abstractmethod
    def get_params(self):
        """Get the parameters of the model."""
        pass

    @abstractmethod
    def set_params(self, **params):
        """Set the parameters of the model."""
        pass
