# Importa as bibliotecas necessárias
import pandas as pd  # O 'pandas' é usada para manipulação de dados, como ler e manipular tabelas (DataFrames).
from sklearn.preprocessing import LabelEncoder, StandardScaler  # 'LabelEncoder' é usado para converter variáveis categóricas em números.
                                                                # 'StandardScaler' para padronizar as variáveis.

def load_data(filepath):
    """Carrega os dados do arquivo Excel."""
    # Carrega os dados do arquivo Excel.
    data = pd.read_excel(filepath)
    
    # Verifica se há valores nulos na base de dados.
    if data.isnull().any().any():
        print("Aviso: Existem valores nulos nos dados.")
        # Se houver valores nulos remove as linhas com valores nulos
        data = data.dropna()
    return data  # Retorna a base de dados sem valores nulos.

def preprocess_data(data):
    """Processa os dados: codifica variáveis categóricas e padroniza os dados."""
    label_encoders = {}  # Dicionário para armazenar os codificadores.

    # Codifica variávei transforma texto em números.
    for column in data.select_dtypes(include=['object']).columns:
        if column != 'Preference':  # A coluna 'Preference' e não queremos codificár.
            le = LabelEncoder()  # Cria um codificador para as variáveis.
            # Aplica o codificador nas colunas e substitui os valores por numeros.
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le  # Armazena o codificador.

    # Codificação da coluna 'Preference' variável que usei para previcao.
    le_preference = LabelEncoder()  # Cria um codificador para a coluna 'Preference'
    # Aplica o codificador à coluna 'Preference' para transformá-la em números.
    data['Preference'] = le_preference.fit_transform(data['Preference'])
    label_encoders['Preference'] = le_preference  # Armazena o codificador 'Preference'

   # Separa as variáveis independentes (X) e a variável dependente (y)
    X = data.drop(columns=['Preference'])  # X são as colunas, exceto 'Preference'
    y = data['Preference']  # y é a variável alvo 'Preference'

    # Padronização dos dados deixando as variáveis com média 0 e desvio padrão 1
    scaler = StandardScaler()  # Cria um padronizardor para os dados
    X = scaler.fit_transform(X)  # Aplica a padronização nas variáveis independentes (X).

    # Retorna os dados processados.
    return X, y, scaler, label_encoders
