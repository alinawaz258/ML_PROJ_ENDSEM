"""
Model Exporter Utility.

Handles model serialization and conversion (ONNX / lightweight formats)
for edge deployment.

Usage
-----
    from src.deployment.exporter import ModelExporter
    exporter = ModelExporter(cfg)
    exporter.run()
"""

import os
from pathlib import Path
from typing import Any, Dict

from src.utils.logger import get_logger
from src.utils.common import get_project_root, save_json

logger = get_logger(__name__)


class ModelExporter:
    """Model Exporter for edge deployment and model auditing."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.root = get_project_root()
        self.models_dir = self.root / cfg["paths"]["models_dir"]
        self.reports_dir = self.root / cfg["paths"]["reports_dir"]

    def export_summary(self) -> Dict[str, Any]:
        """Generate a summary of all saved model artifacts and file sizes."""
        summary = {"models": [], "total_size_mb": 0.0}
        total_bytes = 0

        if self.models_dir.exists():
            for f in sorted(self.models_dir.glob("*")):
                if f.is_file():
                    size_mb = f.stat().st_size / (1024 * 1024)
                    total_bytes += f.stat().st_size
                    summary["models"].append({
                        "name": f.name,
                        "extension": f.suffix,
                        "size_mb": round(size_mb, 2),
                    })

        summary["total_size_mb"] = round(total_bytes / (1024 * 1024), 2)
        logger.info("Model export summary generated: %d models, total %.2f MB",
                    len(summary["models"]), summary["total_size_mb"])
        return summary

    def run(self) -> Dict[str, Any]:
        """Run export summary generation."""
        summary = self.export_summary()
        save_json(summary, str(self.reports_dir / "exported_models_summary.json"))
        return summary
