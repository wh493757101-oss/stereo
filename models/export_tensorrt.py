"""Export YOLO segmentation models to TensorRT / ONNX.

Supports two models:
  - Model A: standard 3-channel grayscale image (fast segmentation)
  - Model B: standard 3-channel [gray, polar, gray] image (material classification)
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def export_model(model_path: str, imgsz: int = 640, half: bool = True,
                 fmt: str = "engine", device: str = "0") -> Path:
    model = YOLO(model_path)
    out = model.export(format=fmt, imgsz=imgsz, half=half, device=device)
    return Path(out)


def main():
    parser = argparse.ArgumentParser(description="Export YOLO seg models")
    parser.add_argument("--model-a", help="Stage-A model path (.pt)")
    parser.add_argument("--model-b", help="Stage-B polar3 model path (.pt)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", default=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--format", default="engine", choices=["engine", "onnx"])
    args = parser.parse_args()

    if args.model_a:
        p = export_model(args.model_a, args.imgsz, args.half, args.format, args.device)
        print(f"Model A exported: {p}")
    if args.model_b:
        p = export_model(args.model_b, args.imgsz, args.half, args.format, args.device)
        print(f"Model B exported: {p}")
    if not args.model_a and not args.model_b:
        print("Specify --model-a and/or --model-b")


if __name__ == "__main__":
    main()
