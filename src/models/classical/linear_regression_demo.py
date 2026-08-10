import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import label_binarize
from src.utils.logger import get_logger
from src.utils.common import timer, save_json
import numpy as np
import os

logger = get_logger(__name__)

class LinearRegressionDemo:
    def __init__(self, cfg):
        self.cfg = cfg
        
    @timer
    def run_demo(self, X_train, y_train, X_test, y_test, class_names, save_dir):
        logger.info("Running LinearRegressionDemo...")
        print("Linear Regression Demo: Frame classification as multi-output regression.")
        print("Educational Note: This is suboptimal because linear regression can predict values outside [0,1], doesn't output probabilities directly, and is sensitive to outliers. Logistic Regression is better suited.")
        
        y_bin_train = label_binarize(y_train, classes=np.arange(len(class_names)))
        y_bin_test = label_binarize(y_test, classes=np.arange(len(class_names)))
        
        model = LinearRegression()
        model.fit(X_train, y_bin_train)
        
        preds_cont = model.predict(X_test)
        preds_class = np.argmax(preds_cont, axis=1)
        
        mse = float(mean_squared_error(y_bin_test, preds_cont))
        r2 = float(r2_score(y_bin_test, preds_cont))
        acc = float(accuracy_score(y_test, preds_class))
        
        results = {'mse': mse, 'r2': r2, 'accuracy': acc}
        
        # Plot
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=preds_class, alpha=0.3)
        plt.xlabel("Actual Class Index")
        plt.ylabel("Predicted Class Index")
        plt.title("Linear Regression for Classification: Predicted vs Actual (Suboptimal)")
        plt.savefig(os.path.join(save_dir, "linear_regression_demo.png"), dpi=150)
        plt.close()
        
        save_json(results, os.path.join(save_dir, "linear_regression_demo_results.json"))
        return results
