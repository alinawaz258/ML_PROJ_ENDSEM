"""
Main pipeline orchestrator for the Tomato Leaf Disease Detection project.

This script manages the end-to-end execution of the project phases:
1. Dataset Preparation
2. Image Preprocessing
3. Feature Extraction
4. Classical ML Training
5. Academic ML Demos (Clustering, Sequential)
6. Deep Learning Training
7. Evaluation & Comparison
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")

from src.utils.logger import get_logger
from src.utils.common import load_config, get_project_root, ensure_dirs
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def run_phase(phase_name: str, func: callable) -> None:
    """
    Run a specific pipeline phase with error handling and logging.

    Parameters
    ----------
    phase_name : str
        Name of the phase.
    func : callable
        Function to execute for the phase.
    """
    logger.info("=" * 60)
    logger.info("PHASE: %s", phase_name)
    logger.info("=" * 60)
    try:
        func()
        logger.info("Phase %s completed successfully.", phase_name)
    except Exception as e:
        logger.error("Phase %s failed: %s", phase_name, e, exc_info=True)
        raise


def phase_dataset() -> None:
    """Execute dataset preparation phase."""
    from src.data.dataset import DatasetManager
    cfg = load_config()
    dm = DatasetManager(cfg)
    dm.run()


def phase_preprocessing() -> None:
    """Execute image preprocessing phase."""
    from src.preprocessing.preprocessor import Preprocessor
    cfg = load_config()
    prep = Preprocessor(cfg)
    prep.run()


def phase_features() -> None:
    """Run the feature extraction pipeline by executing scripts/run_features.py logic."""
    root = get_project_root()
    subprocess.run([sys.executable, str(root / "scripts" / "run_features.py")], check=True)


def phase_classical_ml() -> None:
    """Run classical ML training."""
    root = get_project_root()
    subprocess.run([sys.executable, str(root / "scripts" / "run_classical_ml.py")], check=True)


def phase_demos() -> None:
    """Run academic ML demos."""
    root = get_project_root()
    subprocess.run([sys.executable, str(root / "scripts" / "run_demos.py")], check=True)


def phase_deep_learning() -> None:
    """Run deep learning training."""
    root = get_project_root()
    subprocess.run([sys.executable, str(root / "scripts" / "run_deep_learning.py")], check=True)


def phase_evaluation() -> None:
    """Run evaluation and comparison."""
    root = get_project_root()
    subprocess.run([sys.executable, str(root / "scripts" / "run_evaluation.py")], check=True)


def main() -> None:
    """Run the complete pipeline."""
    cfg = load_config()
    set_global_seed(cfg["dataset"]["random_state"])
    ensure_dirs(cfg)
    
    logger.info("Starting Tomato Leaf Disease Detection Pipeline")
    logger.info("Project: %s v%s", cfg["project"]["name"], cfg["project"]["version"])
    
    phases = [
        ("1. Dataset Preparation", phase_dataset),
        ("2. Image Preprocessing", phase_preprocessing),
        ("3. Feature Extraction", phase_features),
        ("4. Classical ML Training", phase_classical_ml),
        ("5. Academic ML Demos", phase_demos),
        ("6. Deep Learning Training", phase_deep_learning),
        ("7. Evaluation & Comparison", phase_evaluation),
    ]
    
    for name, func in phases:
        run_phase(name, func)
    
    logger.info("=" * 60)
    logger.info("ALL PHASES COMPLETE")
    logger.info("=" * 60)
    print("\nPipeline finished successfully!")
    print("Results: artifacts/reports/")
    print("Figures: artifacts/figures/")
    print("Models: artifacts/models/")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Tomato Leaf Disease Detection Pipeline")
        parser.add_argument("--phase", type=int, default=0, help="Run specific phase (1-7). 0 = run all.")
        args = parser.parse_args()
        
        cfg = load_config()
        set_global_seed(cfg["dataset"]["random_state"])
        ensure_dirs(cfg)
        
        if args.phase == 0:
            main()
        else:
            phase_map = {
                1: ("Dataset Preparation", phase_dataset),
                2: ("Image Preprocessing", phase_preprocessing),
                3: ("Feature Extraction", phase_features),
                4: ("Classical ML Training", phase_classical_ml),
                5: ("Academic ML Demos", phase_demos),
                6: ("Deep Learning Training", phase_deep_learning),
                7: ("Evaluation & Comparison", phase_evaluation),
            }
            if args.phase in phase_map:
                name, func = phase_map[args.phase]
                run_phase(name, func)
            else:
                print(f"Invalid phase: {args.phase}. Use 1-7.")
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        raise
