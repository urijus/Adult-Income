
def train(dataloader, optimizer, model, criterion, tarepochs):
    for epoch in range(epochs):  # Adjust the number of epochs
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, target)  # Assuming you have a target variable
            loss.backward()
            optimizer.step()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test, metrics_list):
    predictions = model.predict(X_test)
    results = {}

    if 'accuracy' in metrics_list:
        results['accuracy'] = accuracy_score(y_test, predictions)
    if 'precision' in metrics_list:
        results['precision'] = precision_score(y_test, predictions)
    if 'recall' in metrics_list:
        results['recall'] = recall_score(y_test, predictions)
    if 'f1_score' in metrics_list:
        results['f1_score'] = f1_score(y_test, predictions)

    return results
