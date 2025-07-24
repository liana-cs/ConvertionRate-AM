import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from preprocessing import DataPreprocessor
import joblib
import os
import shutil
import warnings
warnings.filterwarnings('ignore')

class MLModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.preprocessor = None
        self.models = {
            'Ridge': Ridge(random_state=42),
            'DecisionTree': DecisionTreeRegressor(random_state=42),
            'MLP': MLPRegressor(random_state=42, max_iter=1000),
            'RandomForest': RandomForestRegressor(random_state=42),
            'ElasticNet': ElasticNet(random_state=42),
            'KNN': KNeighborsRegressor()
        }

        self.param_combinations = {
            'Ridge': [
                {'alpha': 1.0, 'solver': 'auto', 'fit_intercept': True, 'tol': 1e-4},
                {'alpha': 10.0, 'solver': 'svd', 'fit_intercept': False, 'tol': 1e-3},
                {'alpha': 100.0, 'solver': 'cholesky', 'fit_intercept': True, 'tol': 1e-2},
                {'alpha': 0.1, 'solver': 'lsqr', 'fit_intercept': False, 'tol': 1e-1},
                {'alpha': 1000.0, 'solver': 'sparse_cg', 'fit_intercept': True, 'tol': 1.0}
            ],
            'DecisionTree': [
                {'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'squared_error'},
                {'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'friedman_mse'},
                {'max_depth': 15, 'min_samples_split': 10, 'min_samples_leaf': 4, 'criterion': 'absolute_error'},
                {'max_depth': None, 'min_samples_split': 20, 'min_samples_leaf': 6, 'criterion': 'squared_error'},
                {'max_depth': 7, 'min_samples_split': 15, 'min_samples_leaf': 8, 'criterion': 'friedman_mse'}
            ],
            'MLP': [
                {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'learning_rate': 'constant'},
                {'hidden_layer_sizes': (50, 50), 'activation': 'tanh', 'solver': 'lbfgs', 'learning_rate': 'invscaling'},
                {'hidden_layer_sizes': (100, 50), 'activation': 'logistic', 'solver': 'sgd', 'learning_rate': 'adaptive'},
                {'hidden_layer_sizes': (150,), 'activation': 'relu', 'solver': 'adam', 'learning_rate': 'invscaling'},
                {'hidden_layer_sizes': (200, 100, 50), 'activation': 'tanh', 'solver': 'lbfgs', 'learning_rate': 'constant'}
            ],
            'RandomForest': [
                {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1},
                {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2},
                {'n_estimators': 300, 'max_depth': None, 'min_samples_split': 10, 'min_samples_leaf': 4},
                {'n_estimators': 150, 'max_depth': 5, 'min_samples_split': 15, 'min_samples_leaf': 6},
                {'n_estimators': 500, 'max_depth': 20, 'min_samples_split': 20, 'min_samples_leaf': 8}
            ],
            'ElasticNet': [
                {'alpha': 1.0, 'l1_ratio': 0.5, 'fit_intercept': True, 'tol': 1e-4},
                {'alpha': 0.1, 'l1_ratio': 0.3, 'fit_intercept': False, 'tol': 1e-3},
                {'alpha': 10.0, 'l1_ratio': 0.7, 'fit_intercept': True, 'tol': 1e-2},
                {'alpha': 100.0, 'l1_ratio': 0.1, 'fit_intercept': False, 'tol': 1e-1},
                {'alpha': 0.01, 'l1_ratio': 0.9, 'fit_intercept': True, 'tol': 1.0}
            ],
            'KNN': [
                {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'auto', 'p': 2},
                {'n_neighbors': 3, 'weights': 'distance', 'algorithm': 'ball_tree', 'p': 1},
                {'n_neighbors': 7, 'weights': 'uniform', 'algorithm': 'kd_tree', 'p': 2},
                {'n_neighbors': 10, 'weights': 'distance', 'algorithm': 'brute', 'p': 1},
                {'n_neighbors': 15, 'weights': 'uniform', 'algorithm': 'auto', 'p': 2}
            ]
        }
    
    def load_and_prepare_data(self):

        self.preprocessor = DataPreprocessor(target_column='Total_Conversion')
        
        df = self.preprocessor.load_data(self.data_path)
        X, y = self.preprocessor.analyze_data(df)
        
        self.preprocessor.create_preprocessor()
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = self.preprocessor.fit_transform_data(
            X, y, test_size=0.3, random_state=1
        )
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    
    def calculate_metrics(self, y_true, y_pred):

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2
    
    def train_and_evaluate_model(self, model_name, model, param_combinations, X_train, X_test, y_train, y_test):
        print(f"\nTreinando {model_name} com {len(param_combinations)} combinações de hiperparâmetros...")
        
        best_result = None
        best_rmse = float('inf')
        
        for i, params in enumerate(param_combinations, 1):
            print(f"  Treinando combinação {i}/5...")
            
            with mlflow.start_run(run_name=f"{model_name}_run_{i}"):

                current_model = model.set_params(**params)

                if model_name == 'SVR':
                    current_model.fit(X_train, y_train)
                    y_pred = current_model.predict(X_test)
                else:
                    current_model.fit(X_train, y_train)
                    y_pred = current_model.predict(X_test)

                rmse, mae, r2 = self.calculate_metrics(y_test, y_pred)

                mlflow.log_params(params)

                mlflow.log_metrics({
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'combination_number': i
                })

                mlflow.sklearn.log_model(
                    current_model, 
                    "model",
                    registered_model_name=f"{model_name}_model_run_{i}"
                )
                
                print(f"    Combinação {i} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_result = {
                        'model_name': model_name,
                        'model': current_model,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'run_id': mlflow.active_run().info.run_id,
                        'best_params': params,
                        'combination_number': i
                    }
        
        print(f"{model_name} - Melhor resultado da combinação {best_result['combination_number']}")
        print(f"   RMSE: {best_result['rmse']:.4f}, MAE: {best_result['mae']:.4f}, R2: {best_result['r2']:.4f}")
        
        return best_result
    
    def train_all_models(self):
 
        mlflow.set_experiment("ML_Regression_Experiment")

        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = self.load_and_prepare_data()
        
        results = []

        for model_name in self.models.keys():
            model = self.models[model_name]
            param_combinations = self.param_combinations[model_name]

            if model_name in ['MLP', 'KNN']:
                result = self.train_and_evaluate_model(
                    model_name, model, param_combinations, 
                    X_train_scaled, X_test_scaled, y_train, y_test
                )
            else:
                result = self.train_and_evaluate_model(
                    model_name, model, param_combinations, 
                    X_train, X_test, y_train, y_test
                )
            
            results.append(result)

        best_result = min(results, key=lambda x: x['rmse'])
        
        print(f"\nMelhor modelo: {best_result['model_name']}")
        print(f"RMSE: {best_result['rmse']:.4f}")
        print(f"MAE: {best_result['mae']:.4f}")
        print(f"R2: {best_result['r2']:.4f}")
        
        return best_result, results
    
    def save_best_model(self, best_result):
        print(f"\nSalvando melhor modelo ({best_result['model_name']})...")

        model_dir = "model_service/best_model"
        if os.path.exists(model_dir):
            print(f"Removendo diretório existente: {model_dir}")
            shutil.rmtree(model_dir)

        os.makedirs(model_dir, exist_ok=True)

        mlflow.sklearn.save_model(
            best_result['model'], 
            model_dir
        )

        self.preprocessor.save_preprocessor(model_dir)

        model_info = {
            'model_name': best_result['model_name'],
            'rmse': best_result['rmse'],
            'mae': best_result['mae'],
            'r2': best_result['r2'],
            'run_id': best_result['run_id'],
            'preprocessing_summary': self.preprocessor.get_preprocessing_summary()
        }
        
        import json
        with open(f"{model_dir}/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        print("Modelo e preprocessor salvos com sucesso!")

def main():

    data_path = "data/KAG_conversion_data.csv"
    if not os.path.exists(data_path):
        print(f"Erro: Dataset não encontrado em {data_path}")
        return

    trainer = MLModelTrainer(data_path)
    best_result, all_results = trainer.train_all_models()
    
    trainer.save_best_model(best_result)
    
    print("\n Treinamento concluído!")
    print("Execute 'mlflow ui' para visualizar os experimentos")

if __name__ == "__main__":
    main()