import torch
from torch.utils.data import TensorDataset, Dataset
from data.processors import CSVProcessor, ExcelProcessor

class AdultDataset(Dataset):
    def __init__(self, file_path, file_type = 'csv', target_column=None, na_values = None, **kwargs):

        if file_type not in ['csv', 'excel']:
            raise ValueError(f"File_type '{file_type}' not supported. Choose from 'csv' or 'excel'.")

        if target_column is None:
            raise ValueError("The 'target_column' parameter must be specified.")

        processor_map = {
            'csv': CSVProcessor,
            'excel': ExcelProcessor
        }
        
        self.na_values = na_values
        self.processor = processor_map[file_type](file_path)
        self._load_data()
        
        #Preprocess if neede, if no no processing is aplplied
        train_combined, test_combined = self._preprocess(target_column=target_column, **kwargs) 
        self.X_train, self.y_train = self._split_features_and_target(train_combined, target_column)
        self.X_test, self.y_test = self._split_features_and_target(test_combined, target_column)

        # Convert to PyTorch datasets
        self.train_dataset = self._to_tensor_dataset(self.X_train, self.y_train)
        self.test_dataset = self._to_tensor_dataset(self.X_test, self.y_test)

    def _load_data(self):
        """Load data using the appropriate processor."""
        return self.processor.load_data(na_values = self.na_values)
    
    def _split_features_and_target(self, combined_df, target_column):
        """Separate features and target from the combined DataFrame."""
        target_columns = [col for col in combined_df.columns if col.startswith(target_column)]
        X = combined_df.drop(columns=target_columns)
        y = combined_df[target_columns]
        return X, y

    def _preprocess(
        self,
        target_column,
        dropna=True,
        encode_target=True,
        encode_type='LabelEncoder',
        test_size=0.3,
        random_state=42,
        num_scaler='MinMaxScaler',
        cat_encoder='TargetEncoder',
        cardinality_threshold=20,
        encode_cat=False,
        scale_num=False,
        feature_trh = False,
        **kwargs
    ):
    
        # Collect all parameters into a dictionary
        preprocess_params = {
            'dropna': dropna,
            'encode_target': encode_target,
            'encode_type': encode_type,
            'test_size': test_size,
            'random_state': random_state,
            'num_scaler': num_scaler,
            'cat_encoder': cat_encoder,
            'cardinality_threshold': cardinality_threshold,
            'encode_cat': encode_cat,
            'scale_num': scale_num,
            'feature_trh': feature_trh
        }

        # Update parameters with any provided extra keyword arguments
        preprocess_params.update(kwargs)

        # Preprocess using the processor's preprocess method
        return self.processor.preprocess(target_column=target_column, **preprocess_params)
    
    def _to_tensor_dataset(self, X, y):
        """Convert features and target data to a PyTorch TensorDataset."""
        X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32)
        y_tensor = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.float32)

        return TensorDataset(X_tensor, y_tensor)

    def get_train_dataset(self, split = False):
        """Return the training dataset."""
    
        if split:
            X_tensor, y_tensor = self.train_dataset.tensors
            return X_tensor, y_tensor
        else:
            return self.train_dataset

    def get_test_dataset(self, split = False):
        """Return the testing dataset."""
        if split:
            X_tensor, y_tensor = self.test_dataset.tensors
            return X_tensor, y_tensor
        else:
            return self.test_dataset

    def __len__(self):
        """Return the number of samples in the training dataset."""
        return len(self.train_dataset)

    def __getitem__(self, idx):
        """Retrieve a sample from the training dataset."""
        sample = self.train_dataset[idx]
        return sample