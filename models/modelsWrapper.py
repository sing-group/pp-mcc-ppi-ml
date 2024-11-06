from sklearn.base import BaseEstimator, ClassifierMixin
from cuml.neighbors import KNeighborsClassifier as cuMLKNeighborsClassifier
class cuMLKNNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, algorithm='auto'):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        # Intenta importar cuML al momento de la creación de la instancia
        try:
            self.model_ = cuMLKNeighborsClassifier(n_neighbors=self.n_neighbors, algorithm=self.algorithm)
        except ImportError:
            raise ImportError("cuML no está instalado. Por favor, instale cuML para usar esta funcionalidad.")
        
    def fit(self, X, y):
        # Ajusta el modelo a los datos
        self.model_.fit(X, y)
        return self
    
    def predict(self, X):
        # Realizar predicciones con el modelo ajustado
        return self.model_.predict(X)
    
    def predict_proba(self, X):
        # Devuelve las probabilidades de las clases
        return self.model_.predict_proba(X)
    
    def score(self, X, y):
        # Devuelve la precisión del clasificador en el conjunto dado
        return self.model_.score(X, y)
    
    def get_params(self, deep=True):
        # Devuelve los parámetros del estimador para GridSearchCV
        return {"n_neighbors": self.n_neighbors, "algorithm": self.algorithm}
    
    def set_params(self, **parameters):
        # Establece los parámetros del estimador
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.model_ = cuMLKNeighborsClassifier(n_neighbors=self.n_neighbors, algorithm=self.algorithm)
        return self