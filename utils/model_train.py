from sklearn.neural_network import MLPClassifier  # Importa o MLP da biblioteca sklearn
from sklearn.metrics import accuracy_score, confusion_matrix  # Importa métricas para avaliação
from tensorflow.keras.models import Sequential  # Importa a classe Sequential para criar o modelo
from tensorflow.keras.layers import Dense  # Importa a camada Dense para redes neurais
import numpy as np  # Biblioteca para manipulação de arrays
import joblib  # Para salvar e carregar modelos

# Função que treina e salva o modelo
def train_and_save_model(X_train, y_train, X_test, y_test, label_encoders):
    # Cria o modelo de Rede Neural MLP usando o sklearn
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 50),  # Define o número de neurônios nas camadas ocultas
                        activation='relu',  # Função de ativação relu para introduzir não-linearidade
                        solver='adam',  # Algoritmo de otimização
                        max_iter=2000,  # Número máximo de iterações para o treino
                        random_state=42)
    
    # Treina o modelo MLP com os dados de treinamento
    mlp.fit(X_train, y_train)

    # Faz previsões usando o modelo treinado do conjunto de treinamento
    y_train_pred = mlp.predict(X_train)
    # Faz previsões usando o modelo treinado do conjunto de teste
    y_test_pred = mlp.predict(X_test)

    # Avalia a acurácia do modelo MLP nos conjuntos de treino e teste
    train_accuracy = accuracy_score(y_train, y_train_pred)  # Acurácia no treino
    test_accuracy = accuracy_score(y_test, y_test_pred)  # Acurácia no teste
    
    # Exibe a acurácia nos conjuntos de treinamento e teste
    print(f"Acurácia no conjunto de treinamento (sklearn): {train_accuracy * 100:.2f}%")
    print(f"Acurácia no conjunto de teste (sklearn): {test_accuracy * 100:.2f}%")

    # Cria uma matriz de confusão para avaliar a performance do modelo
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    print(f"Matriz de Confusão (sklearn):\n{conf_matrix}")

    # Criação do modelo de rede neural utilizando o TensorFlow.
    model_RN = Sequential([
        # Defino o numero de entrada com 10 neurônios
        Dense(10, input_shape=(X_train.shape[1],), activation='relu'),
        # Segunda camada oculta com 10 neurônios
        Dense(10, activation='relu'),
        # Camada de saída com o número de neurônios
        # Utiliza softmax para classificar múltiplas classes
        Dense(len(np.unique(y_train)), activation='softmax')
    ])

    # Compila o modelo
    model_RN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Treina o modelo TensorFlow com o conjunto de treinamento, também passando o conjunto de teste para validação
    model_RN.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)

    # Avalia a performance do modelo nos conjuntos de treinamento e teste
    train_loss, train_accuracy = model_RN.evaluate(X_train, y_train)  # Perda e acurácia do treinamento
    test_loss, test_accuracy = model_RN.evaluate(X_test, y_test)  # Perda e acurácia do teste

    # Exibe a acurácia do modelo de treinamento e do teste
    print(f"Acurácia no conjunto de treino (TensorFlow): {train_accuracy * 100:.2f}%")
    print(f"Acurácia no conjunto de teste (TensorFlow): {test_accuracy * 100:.2f}%")

    # Salva o modelo treinado em um arquivo .h5, que pode ser carregado posteriormente
    model_RN.save('model/modelo_RN.h5')

    # Retorna a acurácia no teste e a matriz de confusão.
    return test_accuracy, conf_matrix
