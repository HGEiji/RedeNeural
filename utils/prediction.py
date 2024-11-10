# Importa as bibliotecas necessárias
import joblib  # carrega o modelo treinado
import numpy as np  # Biblioteca para manipulação de arrays

# Função para carregar o modelo treinado
def load_model():
    """Carrega o modelo treinado salvo em um arquivo"""
    return joblib.load('model/model.joblib')  # Carrega o modelo a partir do arquivo 'model.joblib'

# Função para carregar o scaler (normalizador) e os codificadores de rótulos
def load_scaler_and_encoders():
    """Carrega o scaler e os encoders de rótulos para normalizar e converter variáveis categóricas"""
    scaler = joblib.load('model/scaler.joblib')  # Carrega o scaler que foi usado para padronizar os dados
    label_encoders = joblib.load('model/label_encoders.joblib')  # Carrega os codificadores usados para transformar variáveis categóricas
    return scaler, label_encoders  # Retorna o scaler e os codificadores

# Função para fazer a previsão com base nos dados de entrada
def make_prediction(model, input_data, label_encoders, scaler):
    """Faz a previsão com base nos dados de entrada fornecidos"""
    
    # Processa as variáveis categóricas nas entradas, transformando-as com os label_encoders
    for column, le in label_encoders.items():  # Para cada coluna e codificador
        if column != 'Preference' and column in input_data:  # Não transforma a variável alvo (Preference)
            input_data[column] = le.transform([input_data[column]])[0]  # Transforma a entrada usando o codificador correspondente
    
    # Cria uma lista de características (features) a partir dos dados de entrada, excluindo a coluna 'Preference'
    input_features = [input_data[col] for col in input_data if col != 'Preference']
    
    # Padroniza as entradas (features) usando o scaler carregado
    input_features = scaler.transform([input_features])  # Transforma as features para o mesmo formato utilizado no treinamento
    
    # Faz a previsão com o modelo carregado
    prediction = model.predict(input_features)  # Previsão do modelo com as features normalizadas
    
    # Retorna a previsão (o valor da classe ou categoria prevista)
    return prediction[0]  # Retorna o primeiro (e único) valor da previsão
