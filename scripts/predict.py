import sys
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.common import load_config
from src.deployment.predictor import Predictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Predict tomato leaf disease from an image")
    parser.add_argument("--image", required=True, help="Path to the input image file")
    parser.add_argument("--model", default="efficientnet", choices=["efficientnet", "cnn", "svm", "random_forest"], 
                        help="Model to use for prediction")
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found at {image_path}")
        return
        
    cfg = load_config()
    
    try:
        logger.info(f"Initializing predictor with model: {args.model}")
        predictor = Predictor(cfg=cfg, model_type=args.model)
        
        logger.info(f"Running prediction on {image_path}")
        result = predictor.predict(str(image_path))
        
        print("\n" + "="*50)
        print("PREDICTION RESULT")
        print("="*50)
        print(f"Class:       {result['class_name']}")
        print(f"Confidence:  {result['confidence']:.2%}")
        print("="*50)
        
        print("\nAll Probabilities:")
        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        for cls, prob in sorted_probs:
            if prob > 0.01:
                print(f"  {cls}: {prob:.2%}")
                
    except Exception as e:
        logger.error(f"Failed to run prediction: {e}")

if __name__ == "__main__":
    main()
