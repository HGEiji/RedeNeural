import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_model():
    return joblib.load('model/model.joblib')

def make_prediction(model, input_data):
    df = pd.DataFrame([input_data])
    
    # Codificação dos dados de entrada
    label_encoders = {
        'Gender': LabelEncoder(),
        'Education_Level': LabelEncoder(),
        'Preferred_Activities': LabelEncoder(),
        'Location': LabelEncoder(),
        'Favorite_Season': LabelEncoder(),
    }

    for column, le in label_encoders.items():
        df[column] = le.fit_transform(df[column])

    # Padronização dos dados de entrada
    scaler = StandardScaler()
    df[['Pets', 'Environmental_Concerns']] = scaler.fit_transform(df[['Pets', 'Environmental_Concerns']])
    
    # Previsão
    prediction = model.predict(df)
    return prediction[0]
