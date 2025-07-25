from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
import os
import pandas as pd
import json
import requests
import time
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}
MODEL_SERVICE_URL = os.getenv('MODEL_SERVICE_URL', 'http://127.0.0.1:5001')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('Nenhum arquivo selecionado', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('Nenhum arquivo selecionado', 'error')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            df = pd.read_csv(filepath)
            if df.empty:
                flash('Arquivo CSV está vazio', 'error')
                return redirect(url_for('index'))
            
            logger.info(f"Arquivo carregado: {filename}, Shape: {df.shape}")
            
            return redirect(url_for('train_model', filename=filename))
            
        except Exception as e:
            flash(f'Erro ao ler arquivo CSV: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    flash('Tipo de arquivo não permitido. Use apenas arquivos CSV.', 'error')
    return redirect(url_for('index'))

@app.route('/train/<filename>')
def train_model(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        flash('Arquivo não encontrado', 'error')
        return redirect(url_for('index'))
    
    try:
        with open(filepath, 'rb') as f:
            files = {'file': (filename, f, 'text/csv')}
            response = requests.post(f'{MODEL_SERVICE_URL}/train', files=files, timeout=300)
        
        if response.status_code == 200:
            results = response.json()
            return render_template('results.html', results=results, filename=filename)
        else:
            error_msg = response.json().get('error', 'Erro desconhecido')
            flash(f'Erro no treinamento: {error_msg}', 'error')
            return redirect(url_for('index'))
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Erro na comunicação com model-service: {str(e)}")
        flash('Erro na comunicação com o serviço de modelo', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Erro inesperado: {str(e)}")
        flash(f'Erro inesperado: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)