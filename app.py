from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from utils.data_processing import preprocess_data, load_data
from utils.model_train import train_and_save_model
from utils.prediction import make_prediction, load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'secret_key'

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

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    filepath = request.args.get('filepath')
    data = load_data(filepath)
    X, y = preprocess_data(data)
    accuracy, conf_matrix = train_and_save_model(X, y)
    
    prediction_result = None  # Inicialize a variável para a previsão

    # Verifica se o formulário de previsão foi enviado
    if request.method == 'POST' and 'predict' in request.form:
        input_data = {
            "Gender": request.form['Gender'],
            "Education_Level": request.form['Education_Level'],
            "Preferred_Activities": request.form['Preferred_Activities'],
            "Location": request.form['Location'],
            "Favorite_Season": request.form['Favorite_Season'],
            "Pets": int(request.form['Pets']),
            "Environmental_Concerns": int(request.form['Environmental_Concerns'])
        }
        model = load_model()
        prediction = make_prediction(model, input_data)
        prediction_result = "Gosta de atividade ao ar livre" if prediction == 1 else "Não gosta de atividade ao ar livre"
    
    flash(f"Treinamento completo. Acurácia: {accuracy:.2f}%", "success")
    return render_template('train.html', accuracy=accuracy, conf_matrix=conf_matrix, prediction_result=prediction_result)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists('model'):
        os.makedirs('model')
    app.run(debug=True)
