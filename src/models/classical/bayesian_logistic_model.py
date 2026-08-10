import joblib
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, brier_score_loss
from sklearn.preprocessing import label_binarize
from src.utils.logger import get_logger
from src.utils.common import timer
import numpy as np

logger = get_logger(__name__)

class BayesianLogisticModel:
    """
    Bayesian approximation using multiple BayesianRidge regressors (one-vs-rest).
    This provides a probabilistic interpretation and incorporates prior beliefs about weights.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.models = []
        self.classes = []
        
    @timer
    def train(self, X_train, y_train):
        logger.info("Training BayesianLogisticModel...")
        self.classes = np.unique(y_train)
        y_bin = label_binarize(y_train, classes=self.classes)
        if y_bin.shape[1] == 1:
            y_bin = np.hstack((1-y_bin, y_bin))
        
        self.models = []
        for i in range(len(self.classes)):
            model = BayesianRidge()
            model.fit(X_train, y_bin[:, i])
            self.models.append(model)
            
    def predict_proba(self, X):
        preds = np.column_stack([model.predict(X) for model in self.models])
        # Softmax approximation
        preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
        return preds / np.sum(preds, axis=1, keepdims=True)
        
    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes[np.argmax(probas, axis=1)]
        
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()
        cr = classification_report(y_test, y_pred, zero_division=0)
        
        # Brier score for multi-class approximation
        y_bin = label_binarize(y_test, classes=self.classes)
        if y_bin.shape[1] == 1:
            y_bin = np.hstack((1-y_bin, y_bin))
        brier = float(np.mean([brier_score_loss(y_bin[:, i], y_proba[:, i]) for i in range(len(self.classes))]))
        
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'brier_score': brier,
            'confusion_matrix': cm,
            'classification_report': cr
        }
        
    def save(self, path):
        joblib.dump({'models': self.models, 'classes': self.classes}, path)
        logger.info(f"Model saved to {path}")
        
    def load(self, path):
        data = joblib.load(path)
        self.models = data['models']
        self.classes = data['classes']
        logger.info(f"Model loaded from {path}")
