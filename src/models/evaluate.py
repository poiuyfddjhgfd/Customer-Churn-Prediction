from sklearn.metrics import classification_report,confusion_matrix
def evaluate_model(model,X_test,y_test):
    """
    Evaluate an XGBoost model on test data

    Args:
    model: Trained model.
    X_test:Test features
    y_test:Test labels
    """
    preds=model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))