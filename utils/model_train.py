from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import joblib

# Função de treinamento e salvamento do modelo
def train_and_save_model(X_train, y_train, X_test, y_test, label_encoders):
    # Modelo com sklearn
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 50), 
                        activation='relu', 
                        solver='adam', 
                        max_iter=2000, 
                        random_state=42)
    
    # Treinar o modelo sklearn
    mlp.fit(X_train, y_train)

    # Fazer previsões com o modelo sklearn
    y_train_pred = mlp.predict(X_train)
    y_test_pred = mlp.predict(X_test)

    # Avaliar a acurácia do modelo sklearn
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Acurácia no conjunto de treinamento (sklearn): {train_accuracy * 100:.2f}%")
    print(f"Acurácia no conjunto de teste (sklearn): {test_accuracy * 100:.2f}%")

    # Matriz de confusão sklearn
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    print(f"Matriz de Confusão (sklearn):\n{conf_matrix}")

    # Criar modelo com TensorFlow (Keras)
    model_RN = Sequential([
        Dense(10, input_shape=(X_train.shape[1],), activation='relu'),
        Dense(10, activation='relu'),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])

    # Compilar o modelo
    model_RN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Treinar o modelo TensorFlow
    model_RN.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)

    # Avaliar o modelo TensorFlow
    train_loss, train_accuracy = model_RN.evaluate(X_train, y_train)
    test_loss, test_accuracy = model_RN.evaluate(X_test, y_test)
    print(f"Acurácia no conjunto de treino (TensorFlow): {train_accuracy * 100:.2f}%")
    print(f"Acurácia no conjunto de teste (TensorFlow): {test_accuracy * 100:.2f}%")

    # Salvar o modelo treinado do Keras
    model_RN.save('model/modelo_RN.h5')
    
    # Retornar métricas principais
    return test_accuracy, conf_matrix
