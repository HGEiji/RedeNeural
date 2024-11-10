import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    """Carrega os dados do arquivo Excel."""
    data = pd.read_excel(filepath)
    
    # Verifica se há valores nulos e remove ou preenche (dependendo do caso)
    if data.isnull().any().any():
        print("Aviso: Existem valores nulos nos dados.")
        data = data.dropna()  # Ou você pode optar por preencher com um valor como a média ou moda
    
    return data

def preprocess_data(data):
    """Processa os dados: codifica variáveis categóricas e padroniza os dados."""
    label_encoders = {}
    
    # Codifica variáveis categóricas
    for column in data.select_dtypes(include=['object']).columns:
        if column != 'Preference':  # Não codificar a coluna alvo diretamente
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le

    # Codificar a coluna de 'Preference'
    le_preference = LabelEncoder()
    data['Preference'] = le_preference.fit_transform(data['Preference'])
    label_encoders['Preference'] = le_preference

    # Separação de variáveis independentes e dependentes
    X = data.drop(columns=['Preference'])  # Alvo
    y = data['Preference']

    # Padronização dos dados
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler, label_encoders
