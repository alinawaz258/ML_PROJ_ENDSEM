import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from src.utils.logger import get_logger
from src.utils.common import timer

logger = get_logger(__name__)

class RandomForestModel:
    def __init__(self, cfg):
        self.cfg = cfg
        params = cfg.get('classical_ml', {}).get('random_forest', {'n_estimators': 100})
        self.model = RandomForestClassifier(**params)
        
    @timer
    def train(self, X_train, y_train):
        logger.info(f"Training RandomForestModel...")
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
        
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()
        cr = classification_report(y_test, y_pred, zero_division=0)
        
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'confusion_matrix': cm,
            'classification_report': cr
        }
        
    def save(self, path):
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
        
    def load(self, path):
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
