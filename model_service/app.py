from flask import Flask, request, jsonify
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from train import MLModelTrainer
import tempfile
import shutil
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/train', methods=['POST'])
def train_model():
    try:
        logger.info("Iniciando o treinamento do modelo")

        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name

        logger.info(f"Arquivo temporário criado: {temp_path}")
        
        try:
            logger.info("Iniciando o treinamento dos modelos")
            trainer = MLModelTrainer(temp_path)
            best_result, all_results = trainer.train_all_models()
            trainer.save_best_model(best_result)

            response_data = {
                'best_model': {
                    'model_name': best_result['model_name'],
                    'rmse': best_result['rmse'],
                    'mae': best_result['mae'],
                    'r2': best_result['r2']
                },
                'all_models': [
                    {
                        'model_name': result['model_name'],
                        'rmse': result['rmse'],
                        'mae': result['mae'],
                        'r2': result['r2']
                    } for result in all_results
                ],
                'preprocessing_info': trainer.preprocessor.get_preprocessing_summary()
            }
            
            return jsonify(response_data)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)