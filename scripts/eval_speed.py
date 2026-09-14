"""
推理速度评测脚本。

两种模式：
- model: 单模型 benchmark（PyTorch / ONNX / TensorRT checkpoint）
- pipeline: 端到端 ``DualStageInferenceEngine.from_config`` 流水线 benchmark

用法:
    python scripts/eval_speed.py --model yolov8n-seg.pt
    python scripts/eval_speed.py --model yolov8n-seg.engine --device 0
    python scripts/eval_speed.py --mode pipeline --left l.png --right r.png
    python scripts/eval_speed.py --mode pipeline --left l.png --right r.png \
        --already-rectified --output speed.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def warmup(
    model: Any,
    img: np.ndarray,
    device: str = "cuda",
    imgsz: int = 640,
    n: int = 10,
) -> None:
    if n < 0:
        raise ValueError("warmup count must be zero or greater")
    for _ in range(n):
        model.predict(img, device=device, imgsz=imgsz, verbose=False)


def _summarize(times_ms: np.ndarray) -> dict:
    """Latency statistics shared by both benchmark modes."""
    mean_ms = round(float(times_ms.mean()), 2)
    return {
        "mean_ms": mean_ms,
        "std_ms": round(float(times_ms.std()), 2),
        "min_ms": round(float(times_ms.min()), 2),
        "max_ms": round(float(times_ms.max()), 2),
        "p50_ms": round(float(np.percentile(times_ms, 50)), 2),
        "p95_ms": round(float(np.percentile(times_ms, 95)), 2),
        "fps": round(1000.0 / mean_ms, 1) if mean_ms > 0 else 0.0,
    }


def benchmark(model, img: np.ndarray, device: str = "cuda", imgsz: int = 640,
              n: int = 100) -> dict:
    """跑 n 次推理，返回延迟统计。"""
    if n <= 0:
        raise ValueError("repetition count must be greater than zero")
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(img, device=device, imgsz=imgsz, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)  # ms

    return _summarize(np.array(times))


# ---------------------------------------------------------------------------
# 端到端流水线 benchmark
# ---------------------------------------------------------------------------

def _default_sync() -> Callable[[], None]:
    """Synchronize with the compute stream: CUDA synchronize when a GPU is
    in use, a safe no-op otherwise (CPU-only / torch absent)."""
    try:
        import torch
    except ImportError:
        return lambda: None
    if torch.cuda.is_available():
        return torch.cuda.synchronize
    return lambda: None


def benchmark_pipeline(
    engine: Any,
    left: np.ndarray,
    right: np.ndarray,
    already_rectified: bool = False,
    warmup_n: int = 2,
    n: int = 50,
    sync: Callable[[], None] | None = None,
) -> dict:
    """Benchmark engine.process_frame with synchronized timing.

    Warmup calls are untimed and unsynchronized; every timed frame is
    bracketed by ``sync()`` so queued GPU work cannot leak into the next
    measurement.
    """
    if sync is None:
        sync = _default_sync()
    if warmup_n < 0:
        raise ValueError("warmup count must be zero or greater")
    if n <= 0:
        raise ValueError("repetition count must be greater than zero")

    for _ in range(warmup_n):
        engine.process_frame(left, right, sync_skew_ms=None,
                             already_rectified=already_rectified)

    times = []
    last_instances = 0
    last_valid_depths = 0
    for _ in range(n):
        sync()
        t0 = time.perf_counter()
        instances, depths, _ = engine.process_frame(
            left, right, sync_skew_ms=None, already_rectified=already_rectified
        )
        sync()
        times.append((time.perf_counter() - t0) * 1000)  # ms
        last_instances = len(instances)
        last_valid_depths = sum(1 for depth in depths if depth.get("valid"))

    stats = _summarize(np.array(times))
    stats.update({
        "mode": "pipeline",
        "already_rectified": already_rectified,
        "warmup": warmup_n,
        "repetitions": n,
        "last_frame_instances": last_instances,
        "last_frame_valid_depths": last_valid_depths,
    })
    return stats


def build_pipeline_engine(config_path: Path) -> Any:
    """Build the dual-stage engine from a config file (lazy heavy import)."""
    from gui.inference_engine import DualStageInferenceEngine

    return DualStageInferenceEngine.from_config(config_path)


def run_pipeline_benchmark(
    config_path: Path,
    left_image: Path,
    right_image: Path,
    already_rectified: bool = False,
    warmup: int = 2,
    n: int = 50,
) -> dict:
    """End-to-end pipeline benchmark on one rectified/unrectified pair."""
    left_image = Path(left_image)
    right_image = Path(right_image)
    for image in (left_image, right_image):
        if not image.is_file():
            raise FileNotFoundError(f"Image not found: {image}")

    engine = build_pipeline_engine(config_path)
    left = cv2.imread(str(left_image), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_image), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise ValueError(f"Failed to read stereo pair: {left_image} | {right_image}")

    stats = benchmark_pipeline(
        engine, left, right,
        already_rectified=already_rectified, warmup_n=warmup, n=n,
    )
    stats.update({
        "config": Path(config_path).as_posix(),
        "left_image": str(left_image),
        "right_image": str(right_image),
        "device": getattr(engine, "device", "unknown"),
    })
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO inference speed")
    parser.add_argument(
        "--mode", choices=["model", "pipeline"], default="model",
        help="'model' benchmarks one checkpoint; 'pipeline' benchmarks the "
        "dual-stage inference engine end to end.",
    )
    parser.add_argument("--model", help="模型路径 (.pt / .onnx / .engine)")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--device", default="cuda", help="设备")
    parser.add_argument("--n", type=_positive_int, default=100, help="推理次数")
    parser.add_argument("--warmup", type=_nonnegative_int, default=10, help="预热次数")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"),
        help="pipeline 模式的引擎配置文件",
    )
    parser.add_argument("--left", type=Path, help="pipeline 模式左目图像路径")
    parser.add_argument("--right", type=Path, help="pipeline 模式右目图像路径")
    parser.add_argument(
        "--already-rectified", action="store_true",
        help="pipeline 模式：输入图像已校正，跳过在线校正",
    )
    parser.add_argument("--output", type=Path, help="JSON 报告输出路径")
    args = parser.parse_args(argv)

    if args.mode == "model" and not args.model:
        parser.error("--model is required in model mode")
    if args.mode == "pipeline" and (args.left is None or args.right is None):
        parser.error("--left and --right are required in pipeline mode")
    return args


def _write_json(report: dict, output: Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.mode == "pipeline":
        report = run_pipeline_benchmark(
            args.config, args.left, args.right,
            already_rectified=args.already_rectified,
            warmup=args.warmup, n=args.n,
        )
        print(f"Pipeline: {report['mean_ms']} ms / frame, {report['fps']} FPS")
    else:
        from ultralytics import YOLO

        model = YOLO(args.model)
        img = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

        print(f"Model: {args.model}")
        print(f"Device: {args.device}")
        print(f"Image size: {args.imgsz}x{args.imgsz}")
        print("Warming up...")
        warmup(model, img, device=args.device, imgsz=args.imgsz, n=args.warmup)

        print(f"Running {args.n} inferences...")
        report = benchmark(model, img, device=args.device, imgsz=args.imgsz, n=args.n)
        report["mode"] = "model"
        report["model"] = str(args.model)

        print(f"\n{'='*40}")
        print(f"  Mean latency:  {report['mean_ms']} ms")
        print(f"  Std latency:   {report['std_ms']} ms")
        print(f"  Min latency:   {report['min_ms']} ms")
        print(f"  Max latency:   {report['max_ms']} ms")
        print(f"  P50 latency:   {report['p50_ms']} ms")
        print(f"  P95 latency:   {report['p95_ms']} ms")
        print(f"  FPS:           {report['fps']}")
        print(f"{'='*40}")

    if args.output is not None:
        output = _write_json(report, args.output)
        print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
