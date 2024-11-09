import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    
    data = pd.read_excel(filepath)
    return data

def preprocess_data(data):
   
    input_columns = ['Gender', 'Education_Level', 'Preferred_Activities', 'Location', 'Favorite_Season', 'Pets', 'Environmental_Concerns']
    target_column = 'Preference'
    
    X = data[input_columns].copy()
    y = data[target_column].copy()

    # Codificação de variáveis categóricas
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    # Padronização dos dados
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y
