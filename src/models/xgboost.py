import torch
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer, precision_score, recall_score 
from pathlib import Path

from models.base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, config_path = None):
        super().__init__(config_path)
        self.model = None
        self.params = self.config.get('xgboost', {})['parameters']
        self.param_grid = self.config.get('xgboost', {})['param_grid']
        self.best_params = {}
        self.best_score = None

    def train(self, X_train, y_train, X_test=None, y_test=None, use_best_params = False):
        print('Training an XGBoost model with parameters in model_config.')

        if use_best_params and hasattr(self, 'best_params'):
            print('Training an XGBoost model with the best parameters from hyperparameter tuning.')
            params = self.best_params
        else:
            print('Training an XGBoost model with parameters from model_config.')
            params = self.params

        X_train = X_train.numpy()
        y_train = y_train.numpy()

        self.model = xgb.XGBClassifier(**params)
        eval_set = [(X_test, y_test)] if X_test is not None and y_test is not None else None
        self.model.fit( 
                    X_train, 
                    y_train, 
                    eval_set = eval_set,
                    verbose = False)


    def hyperparameter_tuning(self, X_train, y_train, save_best_model_path=None):
        assert bool(self.param_grid), 'No param_grid defined.'

        if save_best_model_path:
            save_path = Path(save_best_model_path)
            if not save_path.parent.exists():
                raise FileNotFoundError(f"The directory {save_path.parent} does not exist. Please create it before saving the model.")

        xgboost_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')
        grid_search = GridSearchCV(
            estimator=xgboost_model, 
            param_grid=self.param_grid, 
            scoring=make_scorer(roc_auc_score), # Use ROC-AUC as the evaluation metric
            cv=5, # Number of cross-validation folds
            verbose=100, 
            n_jobs=-1 # Use all available cores
        )

        grid_search.fit(X_train, y_train)
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_

        if save_best_model_path:
            best_model = grid_search.best_estimator_
            best_model.save_model(save_best_model_path)
            print(f'Best model saved to {save_best_model_path}')
    

    def evaluate(self, X_test, y_test):
        print('Evaluate the model.')
        X_test = self._transform(X_test)
        y_test = self._transform(y_test)
        
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        
        return {'accuracy': accuracy, 'f1_score': f1, 'precision': precision, 'recall': recall, 'roc_auc': roc_auc}

    def predict(self, X):
        X = self._transform(X)
        return self.model.predict(X)

    def save_model(self, file_path):
        self.model.save_model(file_path)

    def load_model(self, file_path):
        self.model = xgb.XGBClassifier()
        self.model.load_model(file_path)

    def _transform(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        elif isinstance(X, pd.DataFrame):
            X = X.values
        return X

