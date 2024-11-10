# Importando as bibliotecas necessárias
from flask import Flask, request, render_template, redirect, url_for, flash
# Flask é um framework web para criar aplicações. 
# O 'request' é usado para acessar dados de formulários e arquivos enviados pelo usuário.
# O 'render_template' é usado para renderiza arquivos HTML. 
# O 'redirect' redireciona para outra página, 
# O 'flash' exibe mensagens para o usuário.

from werkzeug.utils import secure_filename
# 'secure_filename'  é uma função que garante que os nomes dos arquivos enviados sejam seguros (sem caracteres problemáticos).

import os
import joblib
# 'os' permite interagir com o sistema operacional, como salvar arquivos. 
# 'joblib' é usado para salvar e carregar objetos Python, como modelos treinados.

from utils.data_processing import preprocess_data, load_data
from utils.model_train import train_and_save_model
from utils.prediction import make_prediction, load_model
# Estas importações são funções criados nos arquivos de data_processing, model_train e prediction

# Configurações do Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Pasta onde os arquivos serão armazenados.
app.secret_key = 'secret_key'  # Chave secreta necessária para usar sessões e mensagens flash no Flask.

# Rota para fazer o upload de arquivos (index)
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':  # Verifica se o método da requisição foi enviado.
        file = request.files['file']  # Pega o arquivo enviado.
        if file:  # Se o arquivo foi enviado, ele continua.
            filename = secure_filename(file.filename)  # Garente que o nome do arquivo é seguro.
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Cria o caminho para o arquivo.
            file.save(filepath)  # Salva o arquivo na pasta uploads.
            flash("Arquivo enviado com sucesso!", "success")  # Mostra uma mensagem de sucesso.
            return redirect(url_for('train_model', filepath=filepath))  # Redireciona o para a página de treinamento.
    return render_template('upload.html')  # Exibe o upload.html.

# Rota para treinar o modelo e fazer previsões
@app.route('/train', methods=['GET', 'POST'])
def train_model():
    filepath = request.args.get('filepath')  # Pega o caminho do arquivo da URL.
    data = load_data(filepath)  # Carrega os dados do arquivo enviado.

    # Pré-processamento dos dados separando em variáveis de entrada (X) e saída (y)
    X, y, scaler, label_encoders = preprocess_data(data)

    # Dividindo os dados em conjuntos de treinamento e teste (80% treino e 20% teste)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinando o modelo e salvando o modelo treinado.
    accuracy, conf_matrix = train_and_save_model(X_train, y_train, X_test, y_test, label_encoders)

    # Salvando depois ser usado para fazer previsões.
    joblib.dump(scaler, 'model/scaler.joblib')
    joblib.dump(label_encoders, 'model/label_encoders.joblib')

    prediction_result = None  # Inicia a variável para armazenar o resultado da previsão.

    # Verifica se o formulário de previsão foi enviado.
    if request.method == 'POST' and 'predict' in request.form:
        # Pega os dados de entrada a partir do formulário.
        input_data = {
            "Gender": request.form['Gender'],
            "Education_Level": request.form['Education_Level'],
            "Preferred_Activities": request.form['Preferred_Activities'],
            "Location": request.form['Location'],
            "Favorite_Season": request.form['Favorite_Season'],
            "Pets": int(request.form['Pets']),
            "Environmental_Concerns": int(request.form['Environmental_Concerns'])
        }

        # Carrega o modelo treinado.
        model = load_model()
        
        # Faz a previsão com o modelo carregado, usando os dados de entrada.
        prediction = make_prediction(model, input_data, label_encoders, scaler)
        
        # Exibe o resultado da previsão: se a pessoa gosta de atividades ao ar livre ou não
        prediction_result = "Gosta de atividade ao ar livre" if prediction == 1 else "Não gosta de atividade ao ar livre"

    # Exibe uma mensagem com a acurácia do modelo treinado
    flash(f"Treinamento completo. Acurácia: {accuracy:.2f}%", "success")
    
    # Carrega a página de treinamento, mostrando a acurácia e a matriz de confusão e o do resultado da previsão
    return render_template('train.html', accuracy=accuracy, conf_matrix=conf_matrix, prediction_result=prediction_result)

# Código que garanti que as pastas, uploads e models existem.
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists('model'):
        os.makedirs('model')
    # Inicia o servidor Flask.
    app.run(debug=True)
