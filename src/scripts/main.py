import yaml
from data.processors import CSVProcessor
from models import logistic_regression, decision_tree, random_forest, xgboost_model, neural_network
from utils.config_loader import load_config
from utils.metrics import evaluate_model
from sklearn.model_selection import train_test_split

def main(config):
    # Load data
    processor = CSVProcessor()
    data = processor.load_data(file_path='path/to/adult.csv')
    X, y = processor.preprocess(data, target_column='income')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['training']['test_size'], random_state=config['training']['random_state']
    )

    # Train and evaluate models based on the config
    results = {}
    for model_name, model_config in config['models'].items():
        if model_config['enabled']:
            print(f"Training {model_name}...")
            if model_name == 'logistic_regression':
                model = logistic_regression.train(X_train, y_train, **model_config['parameters'])
            elif model_name == 'decision_tree':
                model = decision_tree.train(X_train, y_train, **model_config['parameters'])
            elif model_name == 'random_forest':
                model = random_forest.train(X_train, y_train, **model_config['parameters'])
            elif model_name == 'xgboost':
                model = xgboost_model.train(X_train, y_train, **model_config['parameters'])
            elif model_name == 'neural_network':
                model = neural_network.train(X_train, y_train, **model_config['parameters'])

            # Evaluate the model
            print(f"Evaluating {model_name}...")
            results[model_name] = evaluate_model(model, X_test, y_test, config['evaluation']['metrics'])

    # Print results
    for model_name, metrics in results.items():
        print(f"\n{model_name} Results:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

if __name__ == '__main__':
    config = load_config('config/config.yaml')
    main(config)
