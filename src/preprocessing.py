# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:

    
    def __init__(self, target_column='Total_Conversion'):
        self.target_column = target_column
        self.preprocessor = None
        self.scaler = None
        self.feature_names = None
        self.categorical_cols = None
        self.numerical_cols = None
        
    def load_data(self, filepath):

        print(f"Carregando dados de: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset não encontrado: {filepath}")
            
        df = pd.read_csv(filepath)
        print(f"Dados carregados: {df.shape}")
        
        return df
    
    def analyze_data(self, df):

        print("\nANÁLISE DOS DADOS:")
        print("-" * 40)
        
        if self.target_column not in df.columns:
            raise ValueError(f"Coluna target '{self.target_column}' não encontrada!")
            
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"Total de features: {len(X.columns)}")
        print(f"Numéricas: {len(self.numerical_cols)} -> {self.numerical_cols}")
        print(f"Categóricas: {len(self.categorical_cols)} -> {self.categorical_cols}")
  
        if self.categorical_cols:
            print("\n Valores categóricos:")
            for col in self.categorical_cols:
                unique_vals = X[col].unique()
                print(f"  {col}: {len(unique_vals)} valores únicos -> {unique_vals[:5]}{'...' if len(unique_vals) > 5 else ''}")

        print(f"\nTarget ({self.target_column}):")
        print(f"  Tipo: {y.dtype}")
        print(f"  Valores únicos: {y.nunique()}")
        print(f"  Média: {y.mean():.4f}")
        print(f"  Min: {y.min():.4f}, Max: {y.max():.4f}")

        missing = df.isnull().sum()
        total_missing = missing.sum()
        if total_missing > 0:
            print(f"\n Valores faltantes encontrados: {total_missing}")
            for col, count in missing.items():
                if count > 0:
                    print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
        else:
            print("\n Nenhum valor faltante!")
            
        return X, y
    
    def create_preprocessor(self):

        print("\n CRIANDO PREPROCESSOR:")
        print("-" * 40)
        
        transformers = []
        
        if self.numerical_cols:
            num_transformer = StandardScaler()
            transformers.append(('num', num_transformer, self.numerical_cols))
            print(f" StandardScaler para {len(self.numerical_cols)} colunas numéricas")
    
        if self.categorical_cols:
            cat_transformer = OneHotEncoder(
                drop='first',  
                sparse_output=False,  
                handle_unknown='ignore'  
            )
            transformers.append(('cat', cat_transformer, self.categorical_cols))
            print(f" OneHotEncoder para {len(self.categorical_cols)} colunas categóricas")
       
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  
        )
        
        print("Preprocessor criado com sucesso!")
        
    def fit_transform_data(self, X, y, test_size=0.3, random_state=1):
        """Aplica o pré-processamento e divide os dados"""
        print("\nAPLICANDO PRÉ-PROCESSAMENTO:")
        print("-" * 40)
        
        X_processed = self.preprocessor.fit_transform(X)
        
        self.feature_names = self._get_feature_names()
        
        print(f"Dados transformados: {X_processed.shape}")
        print(f"Features finais: {len(self.feature_names)}")
       
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        print(f"Divisão dos dados:")
        print(f"  Treino: {X_train.shape[0]} amostras")
        print(f"  Teste: {X_test.shape[0]} amostras")
        print(f"  Média target treino: {y_train.mean():.4f}")
        print(f"  Média target teste: {y_test.mean():.4f}")
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    
    def _get_feature_names(self):
        """Obtém nomes das features após transformação"""
        feature_names = []
        
        if self.numerical_cols:
            feature_names.extend(self.numerical_cols)
       
        if self.categorical_cols:
            cat_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out(self.categorical_cols)
            feature_names.extend(cat_features)
        
        return feature_names
    
    def save_preprocessor(self, save_dir="model_service/best_model"):
        """Salva o preprocessor e scaler"""
        os.makedirs(save_dir, exist_ok=True)
        
        preprocessor_path = os.path.join(save_dir, "preprocessor.pkl")
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"Preprocessor salvo em: {preprocessor_path}")
        
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler salvo em: {scaler_path}")
        
        features_path = os.path.join(save_dir, "feature_names.pkl")
        joblib.dump(self.feature_names, features_path)
        print(f"Feature names salvos em: {features_path}")
    
    def load_preprocessor(self, load_dir="model_service/best_model"):

        preprocessor_path = os.path.join(load_dir, "preprocessor.pkl")
        scaler_path = os.path.join(load_dir, "scaler.pkl")
        features_path = os.path.join(load_dir, "feature_names.pkl")
        
        self.preprocessor = joblib.load(preprocessor_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)
        
        print("Preprocessor carregado com sucesso!")
    
    def transform_new_data(self, X_new, use_scaler=False):

        if self.preprocessor is None:
            raise ValueError("Preprocessor não foi treinado! Use fit_transform_data() primeiro.")
        
        X_processed = self.preprocessor.transform(X_new)
        
        if use_scaler:
            X_processed = self.scaler.transform(X_processed)
        
        return X_processed
    
    def get_preprocessing_summary(self):

        return {
            'numerical_cols': self.numerical_cols,
            'categorical_cols': self.categorical_cols,
            'total_features_original': len(self.numerical_cols) + len(self.categorical_cols) if self.numerical_cols and self.categorical_cols else 0,
            'total_features_processed': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names
        }


def preprocess_data(filepath, target_column='Total_Conversion', test_size=0.3, random_state=1):
  
    preprocessor = DataPreprocessor(target_column=target_column)
    
    df = preprocessor.load_data(filepath)
    X, y = preprocessor.analyze_data(df)
    

    preprocessor.create_preprocessor()
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = preprocessor.fit_transform_data(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return preprocessor, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

if __name__ == "__main__":

    filepath = "data/KAG_conversion_data.csv"
    
    try:
        preprocessor, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = preprocess_data(filepath)
        

        summary = preprocessor.get_preprocessing_summary()
        print(f"\nRESUMO DO PRÉ-PROCESSAMENTO:")
        print(f"  Features originais: {summary['total_features_original']}")
        print(f"  Features após processamento: {summary['total_features_processed']}")
        print(f"  Aumento de dimensionalidade: {summary['total_features_processed'] - summary['total_features_original']}")

        preprocessor.save_preprocessor()
        
        print("\nPré-processamento concluído com sucesso!")
        
    except Exception as e:
        print(f" Erro: {e}")