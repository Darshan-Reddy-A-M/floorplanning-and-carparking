from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ai.model import RoomBoxRegressor


DATA_DIR = Path("data/training/ai_baseline")
OUTPUT_DIR = Path("outputs/room_placement_baseline")

SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
NUM_LABELS = 17

DEVICE = torch.device("cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_plans(split_name: str) -> list[dict]:
    path = DATA_DIR / f"{split_name}.jsonl"

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset split: {path}")

    label_to_id = load_label_mapping()
    plans = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                plan = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}"
                ) from error

            rooms = []
            for room in plan.get("rooms", []):
                label = str(room.get("label", ""))
                bbox = room.get("bbox")

                if label not in label_to_id:
                    continue
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue

                try:
                    bbox = [float(value) for value in bbox]
                except (TypeError, ValueError):
                    continue

                if not np.isfinite(bbox).all():
                    continue

                x1, y1, x2, y2 = bbox
                if x2 <= x1 or y2 <= y1:
                    continue

                rooms.append({"label": label, "bbox": bbox})

            if rooms:
                if "plan_id" not in plan:
                    raise ValueError(
                        f"Missing plan_id in {path}, line {line_number}"
                    )
                plans.append({
                    "plan_id": str(plan["plan_id"]),
                    "rooms": rooms,
                })

    ids = [plan["plan_id"] for plan in plans]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate plan IDs found in the dataset.")

    return plans


def load_label_mapping() -> dict[str, int]:
    path = DATA_DIR / "metadata.json"

    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    return {
        str(label): int(index)
        for label, index in metadata["label_to_id"].items()
    }


def create_examples(
    plans: list[dict],
    label_to_id: dict[str, int],
) -> list[dict]:
    """
    Hide one room at a time and predict its box using the remaining rooms.

    The target room's coordinates are never included in its model input.
    """

    examples = []

    for plan in plans:
        rooms = plan["rooms"]

        for target_index, target_room in enumerate(rooms):
            context = [
                room
                for index, room in enumerate(rooms)
                if index != target_index
            ]

            # This initial model requires at least one context room.
            if not context:
                continue

            target_label = target_room["label"]

            if target_label not in label_to_id:
                raise ValueError(
                    f"Unknown room label '{target_label}' "
                    f"in plan {plan['plan_id']}."
                )

            context_label_ids = []

            for room in context:
                label = room["label"]

                if label not in label_to_id:
                    raise ValueError(
                        f"Unknown context label '{label}' "
                        f"in plan {plan['plan_id']}."
                    )

                context_label_ids.append(label_to_id[label])

            examples.append({
                "plan_id": plan["plan_id"],
                "target_label_id": label_to_id[target_label],
                "target_bbox": target_room["bbox"],
                "context_label_ids": context_label_ids,
                "context_boxes": [
                    room["bbox"] for room in context
                ],
            })

    return examples


class RoomContextDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples
        self.max_context = max(
            len(example["context_label_ids"])
            for example in examples
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        count = len(example["context_label_ids"])
        padding = self.max_context - count

        context_labels = (
            example["context_label_ids"]
            + [0] * padding
        )

        context_boxes = (
            example["context_boxes"]
            + [[0.0, 0.0, 0.0, 0.0]] * padding
        )

        context_mask = [True] * count + [False] * padding

        return {
            "target_label_ids": torch.tensor(
                example["target_label_id"],
                dtype=torch.long,
            ),
            "target_boxes": torch.tensor(
                example["target_bbox"],
                dtype=torch.float32,
            ),
            "context_label_ids": torch.tensor(
                context_labels,
                dtype=torch.long,
            ),
            "context_boxes": torch.tensor(
                context_boxes,
                dtype=torch.float32,
            ),
            "context_mask": torch.tensor(
                context_mask,
                dtype=torch.bool,
            ),
        }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            predictions = model(
                batch["target_label_ids"].to(DEVICE),
                batch["context_label_ids"].to(DEVICE),
                batch["context_boxes"].to(DEVICE),
                batch["context_mask"].to(DEVICE),
            )

            targets = batch["target_boxes"].to(DEVICE)

            loss = loss_function(predictions, targets)

            batch_size = targets.shape[0]
            total_loss += loss.item() * batch_size
            total_iou += calculate_mean_iou(
                predictions, targets
            ) * batch_size
            total_examples += batch_size

    if total_examples == 0:
        return float("nan"), float("nan")

    return (
        total_loss / total_examples,
        total_iou / total_examples,
    )


def calculate_mean_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    intersection_x1 = torch.maximum(
        predictions[:, 0], targets[:, 0]
    )
    intersection_y1 = torch.maximum(
        predictions[:, 1], targets[:, 1]
    )
    intersection_x2 = torch.minimum(
        predictions[:, 2], targets[:, 2]
    )
    intersection_y2 = torch.minimum(
        predictions[:, 3], targets[:, 3]
    )

    intersection_width = (
        intersection_x2 - intersection_x1
    ).clamp(min=0)

    intersection_height = (
        intersection_y2 - intersection_y1
    ).clamp(min=0)

    intersection = intersection_width * intersection_height

    prediction_area = (
        (predictions[:, 2] - predictions[:, 0]).clamp(min=0)
        * (predictions[:, 3] - predictions[:, 1]).clamp(min=0)
    )

    target_area = (
        (targets[:, 2] - targets[:, 0]).clamp(min=0)
        * (targets[:, 3] - targets[:, 1]).clamp(min=0)
    )

    union = prediction_area + target_area - intersection

    iou = intersection / union.clamp(min=1e-8)

    return float(iou.mean().item())


def main() -> None:
    set_seed()

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR.resolve()}"
        )

    label_to_id = load_label_mapping()

    if len(label_to_id) != NUM_LABELS:
        raise ValueError(
            f"Expected {NUM_LABELS} labels, "
            f"found {len(label_to_id)}."
        )

    splits = {}

    for split_name in ("train", "validation", "test"):
        plans = load_plans(split_name)
        examples = create_examples(plans, label_to_id)

        if not examples:
            raise ValueError(
                f"No training examples were generated for "
                f"{split_name}."
            )

        splits[split_name] = {
            "plans": plans,
            "examples": examples,
            "dataset": RoomContextDataset(examples),
        }

        print(
            f"{split_name}: "
            f"{len(plans)} plans, "
            f"{len(examples)} room-placement examples"
        )

    # Check for plan leakage between splits.
    split_ids = {
        name: {plan["plan_id"] for plan in value["plans"]}
        for name, value in splits.items()
    }

    if (
        split_ids["train"] & split_ids["validation"]
        or split_ids["train"] & split_ids["test"]
        or split_ids["validation"] & split_ids["test"]
    ):
        raise ValueError("Plan leakage detected between splits.")

    train_loader = DataLoader(
        splits["train"]["dataset"],
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        splits["validation"]["dataset"],
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        splits["test"]["dataset"],
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = RoomBoxRegressor(
        num_labels=NUM_LABELS,
        embedding_dim=16,
        hidden_dim=128,
    ).to(DEVICE)

    loss_function = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_validation_loss = float("inf")
    best_epoch = 0
    best_model_path = OUTPUT_DIR / "best_model.pt"
    history = []

    print("\nTraining started.")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        seen = 0

        for batch in train_loader:
            optimizer.zero_grad()

            predictions = model(
                batch["target_label_ids"].to(DEVICE),
                batch["context_label_ids"].to(DEVICE),
                batch["context_boxes"].to(DEVICE),
                batch["context_mask"].to(DEVICE),
            )

            targets = batch["target_boxes"].to(DEVICE)

            loss = loss_function(predictions, targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            batch_size = targets.shape[0]
            running_loss += loss.item() * batch_size
            seen += batch_size

        train_loss = running_loss / max(seen, 1)

        validation_loss, validation_iou = evaluate(
            model,
            validation_loader,
            loss_function,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_mean_iou": validation_iou,
        })

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_labels": NUM_LABELS,
                    "embedding_dim": 16,
                    "hidden_dim": 128,
                    "label_to_id": label_to_id,
                    "best_epoch": best_epoch,
                    "validation_loss": best_validation_loss,
                },
                best_model_path,
            )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"train loss={train_loss:.5f} | "
                f"val loss={validation_loss:.5f} | "
                f"val mean IoU={validation_iou:.4f}"
            )

    # Reload the model selected using validation loss.
    checkpoint = torch.load(
        best_model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_iou = evaluate(
        model,
        test_loader,
        loss_function,
    )

    history_path = OUTPUT_DIR / "training_history.json"
    history_path.write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    report = {
        "model": "context_conditioned_room_box_regressor",
        "device": str(DEVICE),
        "seed": SEED,
        "training_plans": len(splits["train"]["plans"]),
        "validation_plans": len(splits["validation"]["plans"]),
        "test_plans": len(splits["test"]["plans"]),
        "training_examples": len(splits["train"]["examples"]),
        "validation_examples": len(splits["validation"]["examples"]),
        "test_examples": len(splits["test"]["examples"]),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "test_loss": test_loss,
        "test_mean_iou": test_iou,
        "warning": (
            "Pilot experiment on a small dataset. Results measure "
            "individual room-box prediction, not complete floor-plan "
            "generation or architectural compliance. Coordinates are "
            "normalized SVG coordinates, not physical measurements."
        ),
    }

    report_path = OUTPUT_DIR / "evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nTraining complete.")
    print("Best epoch:", best_epoch)
    print(f"Test loss: {test_loss:.5f}")
    print(f"Test mean IoU: {test_iou:.4f}")
    print("Saved model:", best_model_path)
    print("Saved history:", history_path)
    print("Saved evaluation:", report_path)


if __name__ == "__main__":
    main()