import joblib
import numpy as np

def load_model():
    return joblib.load('model/model.joblib')

def load_scaler_and_encoders():
    scaler = joblib.load('model/scaler.joblib')
    label_encoders = joblib.load('model/label_encoders.joblib')
    return scaler, label_encoders

def make_prediction(model, input_data, label_encoders, scaler):
    """Faz a previsão com base nos dados de entrada fornecidos"""
    # Transformar as entradas categóricas
    for column, le in label_encoders.items():
        if column != 'Preference' and column in input_data:
            input_data[column] = le.transform([input_data[column]])[0]
    
    # Criar uma estrutura de dados para as entradas (as features)
    input_features = [input_data[col] for col in input_data if col != 'Preference']
    
    # Padronizar as entradas
    input_features = scaler.transform([input_features])
    
    # Fazer a previsão
    prediction = model.predict(input_features)
    
    return prediction[0]