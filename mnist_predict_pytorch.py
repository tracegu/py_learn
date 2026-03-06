# coding: utf-8
"""
MNIST 推理脚本（PyTorch 版）
支持模型评估、可视化和评估报告导出。
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "origin_data"))

from dataset.mnist import load_mnist


class MLP(nn.Module):
    """简单多层感知机：784 -> 100 -> 100 -> 10"""

    def __init__(self, input_size=784, hidden_sizes=None, output_size=10):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [100, 100]

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def parse_args(base_dir: Path):
    parser = argparse.ArgumentParser(description="MNIST PyTorch 推理与评估")
    parser.add_argument("--model-path", type=str, default="mnist_model_pytorch.pth", help="模型权重路径")
    parser.add_argument("--batch-size", type=int, default=256, help="评估批大小")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="推理设备")
    parser.add_argument("--num-samples", type=int, default=10, help="可视化样本数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output-dir", type=str, default=str(base_dir), help="输出目录")
    parser.add_argument("--no-visualize", action="store_true", help="关闭可视化输出")
    parser.add_argument("--save-report", action="store_true", help="保存评估报告 JSON")
    return parser.parse_args()


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def build_model(input_size=784, hidden_sizes=None, output_size=10) -> nn.Module:
    return MLP(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("你指定了 --device cuda，但当前环境不可用 CUDA。")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_weights(model: nn.Module, model_path: Path, device: torch.device) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as exc:
        raise RuntimeError(f"加载模型权重失败: {model_path}\n{exc}") from exc


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    predictions = torch.cat(all_predictions).numpy()
    labels = torch.cat(all_labels).numpy()
    accuracy = float(np.mean(predictions == labels))
    return accuracy, predictions, labels


def compute_per_class_metrics(labels: np.ndarray, predictions: np.ndarray, num_classes=10):
    metrics = []
    for digit in range(num_classes):
        mask = labels == digit
        count = int(np.sum(mask))
        if count == 0:
            acc = 0.0
        else:
            acc = float(np.mean(predictions[mask] == labels[mask]))

        metrics.append(
            {
                "class": digit,
                "accuracy": acc,
                "count": count,
            }
        )
    return metrics


def visualize_predictions(
    model: nn.Module,
    x_test: np.ndarray,
    t_test: np.ndarray,
    device: torch.device,
    num_samples: int,
    seed: int,
    output_path: Path,
):
    if num_samples <= 0:
        print("已跳过可视化（num_samples <= 0）。")
        return

    model.eval()
    rng = np.random.default_rng(seed)
    replace = len(x_test) < num_samples
    indices = rng.choice(len(x_test), size=num_samples, replace=replace)

    fig, axes = plt.subplots(1, num_samples, figsize=(max(12, num_samples * 2.2), 3.2))
    if num_samples == 1:
        axes = [axes]

    with torch.inference_mode():
        for ax, idx in zip(axes, indices):
            image = x_test[idx]
            true_label = int(t_test[idx])

            image_tensor = torch.tensor(image, dtype=torch.float32, device=device).reshape(1, -1)
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred = torch.max(probs, dim=1)

            pred_label = int(pred.item())
            confidence_value = float(confidence.item())

            ax.imshow(image.reshape(28, 28), cmap="gray")
            color = "green" if pred_label == true_label else "red"
            ax.set_title(f"T:{true_label} P:{pred_label} C:{confidence_value:.2f}", color=color, fontsize=9)
            ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)
    print(f"预测可视化已保存到: {output_path}")


def save_report(report_path: Path, report_data: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"评估报告已保存到: {report_path}")


def main() -> int:
    args = parse_args(BASE_DIR)

    if args.batch_size <= 0:
        print("错误: --batch-size 必须大于 0")
        return 1

    if args.num_samples < 0:
        print("错误: --num-samples 不能小于 0")
        return 1

    try:
        device = resolve_device(args.device)
    except Exception as exc:
        print(f"错误: {exc}")
        return 1

    model_path = resolve_path(BASE_DIR, args.model_path)
    output_dir = resolve_path(BASE_DIR, args.output_dir)

    print(f"使用设备: {device}")
    print(f"模型路径: {model_path}")

    model = build_model(input_size=784, hidden_sizes=[100, 100], output_size=10).to(device)
    try:
        load_model_weights(model, model_path, device)
    except Exception as exc:
        print(f"错误: {exc}")
        return 1

    try:
        (_, _), (x_test, t_test) = load_mnist(normalize=True)
    except Exception as exc:
        print(f"错误: 加载 MNIST 数据失败\n{exc}")
        return 1

    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    t_test_tensor = torch.tensor(t_test, dtype=torch.long)
    test_loader = DataLoader(
        TensorDataset(x_test_tensor, t_test_tensor),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("开始评估模型...")
    accuracy, predictions, labels = evaluate_model(model, test_loader, device)
    per_class_metrics = compute_per_class_metrics(labels, predictions, num_classes=10)

    print(f"\n总体测试准确率: {accuracy:.4f}")
    print("\n各数字识别准确率:")
    for item in per_class_metrics:
        print(f"数字 {item['class']}: {item['accuracy']:.4f} ({item['count']} 样本)")

    if not args.no_visualize:
        vis_path = output_dir / "predictions_visualization_pytorch.png"
        visualize_predictions(
            model=model,
            x_test=x_test,
            t_test=t_test,
            device=device,
            num_samples=args.num_samples,
            seed=args.seed,
            output_path=vis_path,
        )

    if args.save_report:
        report_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_path": str(model_path),
            "device": str(device),
            "batch_size": args.batch_size,
            "num_test_samples": int(len(labels)),
            "overall_accuracy": accuracy,
            "per_class_metrics": per_class_metrics,
            "num_samples_for_visualization": args.num_samples,
            "seed": args.seed,
        }
        report_path = output_dir / "mnist_eval_report.json"
        save_report(report_path, report_data)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
