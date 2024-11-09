import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def train_and_save_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    accuracy = accuracy_score(y, model.predict(X)) * 100
    conf_matrix = confusion_matrix(y, model.predict(X))
    joblib.dump(model, 'model/model.joblib')
    return accuracy, conf_matrix
