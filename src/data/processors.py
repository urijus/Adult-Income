"""This file contains the classes to process the dataset."""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pandas.api.types import is_numeric_dtype
from scipy.stats import pointbiserialr, spearmanr
from sklearn.model_selection import train_test_split

from data.utils import TargetTransformer, FeatureTransformer


### Base Processor ###

class BaseProcessor:
    """
    A base class for loading, preprocessing, and transforming datasets.
    
    This class provides a full pipeline for data preparation, including:
    - Loading data (to be implemented by subclasses)
    - Handling missing values
    - Separating features and target
    - Identifying numerical and categorical features
    - Encoding and scaling data
    - Splitting the data into training and testing sets
    
    Subclasses must implement the `load_data` method to load the dataset.
    """
    def __init__(self, file_path):
        """
        Initialize the BaseProcessor with a file path.

        Parameters
        ----------
        file_path : str
            The path to the data file that will be loaded for processing.

        Attributes
        ----------
        df : pd.DataFrame, optional
            The loaded dataset.
        X_train : pd.DataFrame, optional
            Training set features after data splitting.
        X_test : pd.DataFrame, optional
            Test set features after data splitting.
        y_train : pd.Series or pd.DataFrame, optional
            Training set target values.
        y_test : pd.Series or pd.DataFrame, optional
            Test set target values.
        num_features : list, optional
            List of numerical feature column names.
        cat_features : list, optional
            List of categorical feature column names.
        """
        self.file_path = file_path
        self.df = None
        self.len_original_df = 0
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.num_features = None
        self.cat_features = None

    @abstractmethod
    def load_data(self, **kwargs):
        """Abstract method to load data, to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")

    def drop_missing_data(self):
        """Handle missing values by dropping rows with missing data."""
        if self.df is not None:
            self.df = self.df.dropna()
            print(f'Dropped {np.abs(int(len(self.df) - self.len_original_df))} missing values.')

    def separate_features_target(self, target_column = None):
        """Separate features (X) and target (y)."""
        if target_column is None:
            target_column = self.df.columns[-1]

        self.X = self.df.drop(columns=[target_column])
        self.y = self.df[target_column]

    def split_data(self, test_size = 0.3, random_state = 42):
        """Split the dataset into training and testing sets."""
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_encoded_train, self.y_encoded_test = train_test_split(
            self.X, self.y, self.y_encoded, test_size=test_size, random_state=random_state
        )

        assert len(self.X_train) == len(self.y_train), 'Warning;: X_train and y_train lenghts do not match.'
        assert len(self.X_test) == len(self.y_test), 'Warning: X_test and y_test lenghts do not match.'

        print(f'Split dataset with ratio {test_size}.')

    def identify_feature_types(self):
        """Identify numerical and categorical features."""
        self.num_features = self.X.select_dtypes(include=[float, int]).columns.tolist()
        self.cat_features = self.X.columns.difference(self.num_features).tolist()

        print(f'Dataset has {len(self.num_features)} numerical features and {len(self.cat_features)} categorical features.')

    def encode_target(self, target_column, encode_type='LabelEncoder'):
        """Encode the target variable."""
        encoder = TargetTransformer(self.y, encode_type)
        self.y = encoder.encode() #output

        label_encoder = TargetTransformer(self.y, 'LabelEncoder')
        self.y_encoded = label_encoder.encode() #we'll use this to fit the TargetTransformer of categorical values, need to be 1D numerical array.

        print(f"Encoded the target feature y '{target_column}'.")

    def feature_selection(self, feature_trh, train_dataset, test_dataset, target_column):
        '''
        This methods needs revision. It is not operational.
        '''
        df_cols = train_dataset.columns.to_list()
        param = []
        correlation = []
        abs_correlation = []

        combined_dataset = pd.concat([train_dataset, test_dataset], axis=0).reset_index(drop=True)

        for c in df_cols:
            if c != target_column:
                if combined_dataset[c].nunique() <= 2:
                    corr = spearmanr(combined_dataset[target_column], combined_dataset[c])[0]
                else:
                    corr = pointbiserialr(combined_dataset[target_column], combined_dataset[c])[0]
                
                param.append(c)
                correlation.append(corr)
                abs_correlation.append(abs(corr))
        
        param_df = pd.DataFrame({'corr': correlation, 'parameter': param, 'abs_corr': abs_correlation})
        param_df = param_df.sort_values(by=['abs_corr'], ascending = False)
        param_df = param_df.set_index('parameter')

        best_features = list(param_df.index[0:feature_trh].values)
        best_features.append(target_column)

        return train_dataset[best_features], test_dataset[best_features]

    def initialize_transformer(self, num_scaler, cat_encoder, cardinality_threshold, encode_cat, scale_num):
        """Initialize the FeatureTransformer for feature scaling and encoding."""
        return FeatureTransformer(
            num_features=self.num_features,
            cat_features=self.cat_features,
            num_scaler=num_scaler,
            cat_encoder=cat_encoder,
            cardinality_threshold=cardinality_threshold,
            X_train=self.X_train,
            X_test=self.X_test,
            y_encoded_train=self.y_encoded_train,
            encode_cat = encode_cat, 
            scale_num = scale_num
        )

    def apply_transformations(self, transformer):
        """Apply transformations to the training and testing sets."""
        X_train_transformed = transformer.fit_transform()
        X_test_transformed = transformer.transform()

        return X_train_transformed, X_test_transformed

    def to_dataframe(self, y):
        if isinstance(y, pd.Series):
            Y = y.to_frame()
        elif isinstance(y, np.ndarray):
            Y = pd.DataFrame(y)
        else:
            Y = y

        return Y

    def concatenate_target(self, y, X, target_column):
        Y = self.to_dataframe(y)
      
        # Rename the columns to target_1, target_2, ..., etc.
        Y.columns = [f"{target_column}_{i+1}" for i in range(Y.shape[1])]

        # Concatenate the features and renamed target columns
        dataset = pd.concat([X.reset_index(drop=True), Y.reset_index(drop=True)], axis=1)

        return dataset

    def preprocess(
        self, dropna=True, target_column=None, encode_target=True, encode_type='LabelEncoder',
        test_size=0.3, random_state=42, num_scaler='MinMaxScaler', cat_encoder='TargetEncoder',
        cardinality_threshold=20, encode_cat = False, scale_num = False, feature_trh = 5
    ):
        """
        Full data preprocessing pipeline:
        - Handle missing values (optional)
        - Separate features and target
        - Identify numerical and categorical features
        - Encode and scale data
        - Split into training and testing datasets
        """
        # Load data and handle missing values
        if dropna:
            self.drop_missing_data()

        # Separate features and target
        self.separate_features_target(target_column)

        # Encode the target variable if required
        if encode_target:
            self.encode_target(target_column, encode_type)
        else:
            self.y_encoded = self.y

        # Split data into training and testing sets
        self.split_data(test_size=test_size, random_state=random_state)

        # Identify numerical and categorical features
        self.identify_feature_types()

        if encode_cat or scale_num:
            transformer = self.initialize_transformer(num_scaler, cat_encoder, cardinality_threshold, encode_cat, scale_num)
            X_train, X_test = self.apply_transformations(transformer)
            print('Apply scaling or encoding.')

        else:
            X_train, X_test = self.X_train, self.X_test
            print('No scaling or encoding applied.')

        train_dataset = self.concatenate_target(self.y_train, X_train, target_column = target_column)
        test_dataset = self.concatenate_target(self.y_test, X_test, target_column = target_column)

        if feature_trh:
            print(f'Apply feature selection down to {feature_trh} columns')
            print("This methods needs revision. It is not operational.")
            #self.feature_selection(feature_trh, train_dataset, test_dataset, 'income_1')
            pass
            
        return train_dataset, test_dataset
        

### Custom Processors ###

class CSVProcessor(BaseProcessor):
    """
    A processor for loading and handling CSV files.
    Inherits from BaseProcessor and implements the load_data method.
    """
    def load_data(self, **kwargs):
        """Load data from a CSV file."""
        self.df = pd.read_csv(self.file_path, **kwargs)
        self.len_original_df = len(self.df)
        print(f'Loaded a CSV file with {self.len_original_df} rows.')


class ExcelProcessor(BaseProcessor):
    """
    A processor for loading and handling Excel files.
    Inherits from BaseProcessor and implements the load_data method.
    """
    def load_data(self, **kwargs):
        """Load data from an Excel file."""
        self.df = pd.read_excel(self.file_path, **kwargs)
        self.len_original_df = len(self.df)
        print(f'Loaded a Excel file with {self.len_original_df} rows.')

