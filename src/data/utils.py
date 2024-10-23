"""This file contains classes for encoding categorical and numerical data."""

import os
import sys

import numpy as np
import pandas as pd
import category_encoders as ce

from sklearn.preprocessing import (
    OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
)


### Preprocessing Classes ###

class TargetTransformer:
    """
    A class to handle encoding of target variables (y) using various encoding methods.
    
    Supported encoders:
    - LabelEncoder: Encodes target labels with values between 0 and n_classes-1.
    - OneHotEncoder: Converts the target variable into one-hot encoded format.
    - BinaryEncoder: Encodes target labels into binary form (useful for high cardinality features).

    Attributes
        y : pd.Series or np.array
            The target variable to encode.
        encode_type : str
            The type of encoding to apply. Options are 'LabelEncoder', 'OneHotEncoder', and 'BinaryEncoder'.
    
    Methods
        encode():
            Encodes the target variable using the specified encoding method.
        _label_encode():
            Applies label encoding to the target variable.
        _onehot_encode():
            Applies one-hot encoding to the target variable.
        _binary_encode():
            Applies binary encoding to the target variable.
    """
    ENCODERS = {
        'LabelEncoder': '_label_encode',
        'OneHotEncoder': '_onehot_encode',
        'BinaryEncoder': '_binary_encode'
    }

    def __init__(self, y: pd.DataFrame, encode_type: str):
        self.y = y
        self.encode_type = encode_type

    def encode(self) -> np.ndarray:
        """Encode the target variable using the specified encoding type."""
        if self.encode_type not in self.ENCODERS:
            raise ValueError(f"Unknown encode_type '{self.encode_type}'. Choose from 'LabelEncoder', 'OneHotEncoder', or 'BinaryEncoder'.")

        method = getattr(self, self.ENCODERS[self.encode_type])
        return method()

    def _label_encode(self):
        """Apply Label Encoding to the target variable."""
        return LabelEncoder().fit_transform(self.y)
    
    def _onehot_encode(self):
        """Apply One-Hot Encoding to the target variable."""
        onehot_encoder = OneHotEncoder(sparse_output=False)
        return onehot_encoder.fit_transform(self.y.values.reshape(-1, 1))

    def _binary_encode(self):
        """Apply Binary Encoding to the target variable."""
        binary_encoder = ce.BinaryEncoder()
        return binary_encoder.fit_transform(self.y).values


class FeatureTransformer:
    """
    A class for transforming numerical and categorical features with scalers and encoders.
    Supports numerical scaling (MinMaxScaler, StandardScaler) and categorical encoding (TargetEncoder or OneHot).
    """

    NUM_SCALERS = {
        'MinMaxScaler': MinMaxScaler,
        'StandardScaler': StandardScaler
    }

    # For high cardinality categorical features, low cardinality always will be applied one-hot encoding.
    CAT_ENCODERS = {
        'TargetEncoder': ce.TargetEncoder  
    }

    def __init__(self, num_features, cat_features, num_scaler, cat_encoder, cardinality_threshold, X_train, X_test, y_encoded_train, encode_cat, scale_num):
        """
        Parameters:
        ----------
        num_features : list
            List of numerical feature column names.
        cat_features : list
            List of categorical feature column names.
        num_scaler : str
            The scaler type for numerical features ('MinMaxScaler', 'StandardScaler').
        cat_encoder : str
            The encoder type for categorical features ('TargetEncoder').
        cardinality_threshold : int
            The threshold to differentiate between high and low cardinality categorical features.
        X_train : pd.DataFrame
            The training data (features).
        X_test : pd.DataFrame
            The testing data (features).
        y_train : pd.Series or np.array
            The target variable for training data.
        """

        self.num_features = num_features
        self.cat_features = cat_features
        self.cardinality_threshold = cardinality_threshold
        self.X_train = X_train
        self.X_test = X_test
        self.y_encoded_train = y_encoded_train
        self.encode_cat = encode_cat
        self.scale_num = scale_num

        # Validate scaler and encoder input and instantiate
        self.scaler = self._get_scaler(num_scaler)
        self.encoder = self._get_encoder(cat_encoder)

        # Dictionary to store encoders for high and low cardinality features
        self.encoders = {}

    def _get_scaler(self, scaler_type):
        """Helper method to get the numerical scaler."""
        if scaler_type not in self.NUM_SCALERS:
            raise ValueError(f"Unknown num_scaler '{scaler_type}'. Choose from {list(self.NUM_SCALERS.keys())}.")
        return self.NUM_SCALERS[scaler_type]()  # Initialize the scaler

    def _get_encoder(self, encoder_type):
        """Helper method to get the categorical encoder."""
        if encoder_type not in self.CAT_ENCODERS:
            raise ValueError(f"Unknown cat_encoder '{encoder_type}'. Choose from {list(self.CAT_ENCODERS.keys())}.")
        return self.CAT_ENCODERS[encoder_type]  

    def fit_transform(self):
        """
        Fit and transform the training data by encoding categorical features and scaling numerical features.

        Categorical features with high cardinality are target encoded, while low cardinality features are 
        one-hot encoded. Numerical features are scaled using the specified scaler.

        Returns:
        --------
        pd.DataFrame
            The transformed training data with both numerical and categorical features.
        """
        transformed_features = []
        X_train_transformed_num = pd.DataFrame([])
        X_train_transformed_cat = pd.DataFrame([])

        if self.encode_cat:
            # Process categorical features based on cardinality
            for feature in self.cat_features:
                cardinality = self.X_train[feature].nunique()

                if cardinality > self.cardinality_threshold:
                    # Use defined encoder for high cardinality features
                    encoder = self.encoder(cols=[feature])
                    X_train_encoded = encoder.fit_transform(self.X_train[[feature]], self.y_encoded_train)
                    self.encoders[feature] = encoder
                else:
                    onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
                
                    X_train_encoded = pd.DataFrame(
                        onehot_encoder.fit_transform(self.X_train[[feature]]), 
                        columns=onehot_encoder.get_feature_names_out([feature])
                    )
                    X_train_encoded = X_train_encoded.astype(int) 
                    self.encoders[feature] = onehot_encoder

                # Append the transformed DataFrame to the list
                transformed_features.append(X_train_encoded.reset_index(drop=True))

            # Concatenate all transformed categorical features
            X_train_transformed_cat = pd.concat(transformed_features, axis=1).reset_index(drop=True)

        if self.scale_num:
            # Scale numerical features
            X_train_num = self.scaler.fit_transform(self.X_train[self.num_features])
            X_train_transformed_num = pd.DataFrame(X_train_num, columns=self.num_features).reset_index(drop=True)

        # Combine encoded categorical and scaled numerical features based on conditions
        if self.encode_cat and self.scale_num:
            # Combine both numerical and categorical features
            X_train_transformed = pd.concat([X_train_transformed_num, X_train_transformed_cat], axis=1).reset_index(drop=True)
        elif self.encode_cat:
            # Return only encoded categorical features
            X_train_transformed = pd.concat([self.X_train[self.num_features].reset_index(drop=True), X_train_transformed_cat], axis=1).reset_index(drop=True)
        else:
            # Return only scaled numerical features
            X_train_transformed = pd.concat([X_train_transformed_num, self.X_train[self.cat_features].reset_index(drop=True)], axis=1).reset_index(drop=True)

        return X_train_transformed


    def transform(self):
        """
        Transform the test data using the fitted encoders for categorical features 
        and the fitted scaler for numerical features.

        Returns:
        --------
        pd.DataFrame
            The transformed test data with both numerical and categorical features.
        """
        transformed_features = []

        if self.encode_cat:
            # Process categorical features
            for feature in self.cat_features:
                encoder = self.encoders[feature]

                if isinstance(self.encoders[feature], OneHotEncoder):
                    # One-Hot Encoding for low-cardinality features
                    X_test_encoded = pd.DataFrame(
                        encoder.transform(self.X_test[[feature]]), 
                        columns=encoder.get_feature_names_out([feature])
                    )
                else:
                    # Target Encoding for high-cardinality features
                    X_test_encoded = encoder.transform(self.X_test[[feature]])

                transformed_features.append(X_test_encoded.reset_index(drop=True))

            X_test_transformed_cat = pd.concat(transformed_features, axis=1).reset_index(drop=True)

        if self.scale_num:
            # Scale numerical features
            X_test_num = self.scaler.transform(self.X_test[self.num_features])
            X_test_transformed_num = pd.DataFrame(X_test_num, columns=self.num_features)

        # Combine encoded categorical and scaled numerical features based on conditions
        if self.encode_cat and self.scale_num:
            # Combine both numerical and categorical features
            X_test_transformed = pd.concat([X_test_transformed_num, X_test_transformed_cat], axis=1).reset_index(drop=True)
        elif self.encode_cat:
            # Return only encoded categorical features
            X_test_transformed = pd.concat([self.X_test[self.num_features].reset_index(drop=True), X_test_transformed_cat], axis=1).reset_index(drop=True)
        else:
            # Return only scaled numerical features
            X_test_transformed = pd.concat([X_test_transformed_num, self.X_test[self.cat_features].reset_index(drop=True)], axis=1).reset_index(drop=True)

        return X_test_transformed







