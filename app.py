# Importando as bibliotecas necessárias
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import joblib

# Importando funções do diretório 'utils'
from utils.data_processing import preprocess_data, load_data
from utils.model_train import train_and_save_model
from utils.prediction import make_prediction, load_model

# Configurações do Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')  # Caminho absoluto para a pasta 'uploads'
app.secret_key = 'secret_key'  # Chave secreta necessária para usar sessões e mensagens flash no Flask.

# Rota para fazer o upload de arquivos (index)
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            flash("Arquivo enviado com sucesso!", "success")
            return redirect(url_for('train_model', filepath=filepath))
    return render_template('upload.html')


# Função para treinar o modelo
def train_model_process(filepath):
    # Carrega e pré-processa os dados
    data = load_data(filepath)
    X, y, scaler, label_encoders = preprocess_data(data)

    # Divide os dados em conjuntos de treinamento e teste
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treina o modelo e salva o modelo treinado
    accuracy, conf_matrix = train_and_save_model(X_train, y_train, X_test, y_test, label_encoders)

    # Salva o scaler e os codificadores de rótulo para serem usados na predição
    joblib.dump(scaler, 'model/scaler.joblib')
    joblib.dump(label_encoders, 'model/label_encoders.joblib')

    return accuracy, conf_matrix


# Rota para treinar o modelo e exibir resultados
@app.route('/train', methods=['GET', 'POST'])
def train_model():
    filepath = request.args.get('filepath')
    accuracy, conf_matrix = train_model_process(filepath)
    flash(f"Treinamento completo. Acurácia: {accuracy:.2f}%", "success")
    return render_template('train.html', accuracy=accuracy, conf_matrix=conf_matrix)


# Função para fazer a predição
def predict_process(input_data):
    # Carrega o modelo e os pré-processadores
    model = load_model()
    scaler = joblib.load('model/scaler.joblib')
    label_encoders = joblib.load('model/label_encoders.joblib')

    # Realiza a previsão
    prediction = make_prediction(model, input_data, label_encoders, scaler)
    return "Gosta de atividade ao ar livre" if prediction == 1 else "Não gosta de atividade ao ar livre"


# Rota para a página de predição
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_result = None  # Inicializa o resultado da predição como None

    if request.method == 'POST':
        # Coleta os dados do formulário
        input_data = {
            "Gender": request.form['Gender'],
            "Education_Level": request.form['Education_Level'],
            "Preferred_Activities": request.form['Preferred_Activities'],
            "Location": request.form['Location'],
            "Favorite_Season": request.form['Favorite_Season'],
            "Pets": int(request.form['Pets']),
            "Environmental_Concerns": int(request.form['Environmental_Concerns'])
        }

        # Chama o processo de predição
        prediction_result = predict_process(input_data)

    return render_template('predict.html', prediction_result=prediction_result)


# Código que garante que as pastas 'uploads' e 'model' existam.
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists('model'):
        os.makedirs('model')
    # Inicia o servidor Flask.
    app.run(debug=True)