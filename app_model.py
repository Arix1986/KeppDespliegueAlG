


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC
from sklearn.utils import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, precision_score, f1_score,recall_score
import mlflow
import joblib
import scipy.sparse
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"



class MModel():
    def __init__(self, x_train,y_train, x_test,y_test, type_model='all' ,random_state=42):
        np.random.seed(random_state) 
        self.x_train = np.array(x_train) 
        self.y_train = np.array(y_train)
        self.x_test = np.array(x_test)
        self.y_test = np.array(y_test)
        self.random_state = random_state
        self.type=type_model
        self.best_model = None 
        self.best_params = None
        self.class_weights = compute_class_weight(class_weight='balanced',  classes=np.array([0, 1]), y=self.y_train)
        self.class_weight_dict = {0: self.class_weights[0], 1: self.class_weights[1]}
        self.param_grid_TF = {
            'vect__max_df': [0.9],  
            'vect__min_df': [2,3],  
            'vect__max_features': [5000, 9000, 12000, 15000],  
            'vect__ngram_range': [(1,2),(1,3)],
            'clf__C': [0.5,0.7,0.9],
            'clf__solver': ['liblinear', 'saga','sag']
        }
        
        self.param_grid_CV = {
            'vect__max_df': [0.9],  
            'vect__min_df': [2,3],  
            'vect__max_features': [5000, 9000,12000],  
            'vect__ngram_range': [(1,2),(1,3)],
            'clf__C': [0.001,0.01, 0.1,0.5,0.7],
            'clf__kernel':['linear'] 
            
        }
       

        self.models = [
            {
                'name': 'TF-IDF',
                'params': self.param_grid_TF,
                'pipeline': Pipeline([  
                    ('vect', TfidfVectorizer(sublinear_tf=True)),
                   ('clf', LogisticRegression(class_weight=self.class_weight_dict, random_state=self.random_state))
                ])
            },
            {
                'name': 'CountVectorizer',
                'params': self.param_grid_CV,
                'pipeline': Pipeline([
                    ('vect', CountVectorizer()),
                    ('scaler', MaxAbsScaler()),
                    ('clf', SVC(probability=True,class_weight=self.class_weight_dict, random_state=self.random_state))
                ])
            },
            
        ]
    def train_(self):
        best_score = -np.inf
        modelos=self.models if self.type=='all' else [model for model in self.models if model['name']==self.type]
       
        for model in modelos:
            print(f"\n🔹 Entrenando modelo: {model['name']}") 
            b_search = HalvingGridSearchCV(
                model['pipeline'],
                param_grid=model['params'], 
                cv=3,
                factor=10,
                scoring ='precision_weighted',
                n_jobs=1,
                verbose=3,
                random_state=self.random_state
            )

            print(f"\n Entrenando modelo: {model['name']}")       
            b_search.fit(self.x_train, self.y_train)
           
            score = b_search.best_score_
            if score > best_score:
                best_score = score
                self.best_model = b_search.best_estimator_
                self.best_params = b_search.best_params_
            
            print(f" Mejor Accuracy para {model['name']}: {best_score:.4f}")
            joblib.dump(self.best_model, f"best_model_{model['name']}_{best_score:.4f}.pkl")
            self.mlflow_traking()
            self.evaluate_model()
        if self.best_model:
            joblib.dump(self.best_model, f"./models/best_model_{best_score:.4f}.pkl")
            print(f"\n Mejor Accuracy  guardado: {best_score} ")
            
            

    def evaluate_model(self):
        if not self.best_model:
            print("No hay un modelo entrenado. Ejecuta `train_()` primero.")
            return
        
        print("\n Evaluando Mejor Modelo")
        y_pred = self.best_model.predict(self.x_test)

        print("\n Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Negative', 'Positive'],zero_division=1))

        print("\n Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
    
    def mlflow_traking(self):
        y_pred = self.best_model.predict(self.x_test)
        input_example = [self.x_test[0]]
        f1_scores = f1_score(self.y_test, y_pred)
        recall_scores =recall_score(self.y_test, y_pred,zero_division=1)
        presicion=precision_score(self.y_test, y_pred,zero_division=1)
        mlflow.set_experiment("Sentiment_Analysis_v2")
        with mlflow.start_run():
            mlflow.log_param("random_state", self.random_state)
            if self.best_params:
                for key, value in self.best_params.items():
                    mlflow.log_param(key, value)
            else:
                mlflow.log_param("best_params", "No hyperparameters selected")
            mlflow.log_param("class_weights", self.class_weight_dict)
            mlflow.log_param("best_model", self.best_model)
            mlflow.log_metric("f1_score", f1_scores)
            mlflow.log_metric("recall_score", recall_scores)
            mlflow.log_metric("presicion", presicion)
            mlflow.sklearn.log_model(self.best_model, "model",input_example=input_example)
            mlflow.set_tag('Autor', 'Arix')
            mlflow.set_tag('version', '1.0')
            mlflow.end_run()
            