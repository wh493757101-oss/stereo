from models.segmentation import SegmentationModel, Instance
from models.classification import ClassificationModel, ClassificationResult


def __getattr__(name: str):
    # Lazy re-export: import models.export_tensorrt pulls in Ultralytics,
    # which globally patches cv2.imread. Keep it out of ordinary
    # `import models` / `import models.classification` paths.
    if name == "export_model":
        from models.export_tensorrt import export_model

        return export_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
