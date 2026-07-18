"""Run a deterministic CPU smoke-training check for the compact GPT model."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpt.src.model import GPTModel


SMOKE_CONFIG = {
    "vocab_size": 64,
    "context_length": 8,
    "emb_dim": 16,
    "n_heads": 4,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
}


def _loss_value(model: GPTModel, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        loss, _ = model(inputs, targets)
    return float(loss.item())


def run_smoke(seed: int = 42, steps: int = 5) -> dict[str, object]:
    """Train on a fixed synthetic batch and return machine-readable evidence."""
    if steps < 1:
        raise ValueError("steps must be at least 1")

    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = GPTModel(SMOKE_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=0.0)

    inputs = torch.randint(
        low=0,
        high=SMOKE_CONFIG["vocab_size"],
        size=(8, SMOKE_CONFIG["context_length"]),
        device=device,
    )
    targets = torch.roll(inputs, shifts=-1, dims=1)

    initial_loss = _loss_value(model, inputs, targets)
    loss_curve = [initial_loss]

    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(inputs, targets)
        loss.backward()
        optimizer.step()
        loss_curve.append(_loss_value(model, inputs, targets))
        model.train()

    return {
        "run_date": date.today().isoformat(),
        "device": device.type,
        "seed": seed,
        "steps": steps,
        "config": SMOKE_CONFIG,
        "initial_loss": loss_curve[0],
        "final_loss": loss_curve[-1],
        "loss_curve": loss_curve,
        "torch_version": torch.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_smoke(seed=args.seed, steps=args.steps)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
