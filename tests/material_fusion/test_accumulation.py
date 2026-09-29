"""Gradient-accumulation grouping, tail handling and sample weighting.

The accumulation contract: group boundaries are stable (each group covers
exactly ``accum`` consecutive micro-batches except the last, which covers
the remainder and is always stepped); the number of optimizer steps is
``ceil(batches / accum)``; every micro-batch contributes with weight
``n_i / N_group`` (its sample count over the group's sample count), never
an unweighted mean of unequal micro-batches; the epoch loss is the
sample-weighted mean over all processed samples.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from scripts.train_material_fusion import (
    accumulation_groups,
    train_one_epoch,
)


class TinyClassifier(nn.Module):
    """Deterministic classifier: no dropout, no BatchNorm, no randomness."""

    def __init__(self):
        super().__init__()
        self.is_dual = True
        self.fc1 = nn.Linear(3 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, 4)
        self.stages = []

    def forward(self, rgb, aux=None):
        return self.fc2(torch.relu(self.fc1(rgb.flatten(1))))

    def set_stage(self, stage: str) -> None:
        self.stages.append(stage)
        self.train()


class SizedDataset(torch.utils.data.Dataset):
    """Deterministic samples; index-based so two loaders can share content."""

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(1000 + index)
        rgb = torch.rand(3, 8, 8, generator=generator)
        aux = torch.rand(3, 8, 8, generator=generator)
        quality = torch.tensor([1.0, 1.0, 1.0, 0.1])
        return rgb, aux, quality, index % 4


def make_loader(num_samples: int, batch_size: int, shuffle: bool = False):
    return torch.utils.data.DataLoader(
        SizedDataset(num_samples), batch_size=batch_size, shuffle=shuffle
    )


class TestAccumulationGroups:
    @pytest.mark.parametrize(
        "total,accum,expected_steps",
        [
            (1, 1, 1), (2, 1, 2), (5, 1, 5),
            (1, 2, 1), (2, 2, 1), (3, 2, 2), (4, 2, 2), (5, 2, 3),
            (1, 3, 1), (3, 3, 1), (4, 3, 2), (5, 3, 2), (6, 3, 2), (7, 3, 3),
        ],
    )
    def test_group_count_is_ceil(self, total, accum, expected_steps):
        groups = accumulation_groups(total, accum)
        assert len(groups) == expected_steps
        # Groups tile the batch range exactly, with stable boundaries.
        assert groups[0][0] == 0
        assert groups[-1][1] == total
        for (start, end), (next_start, _) in zip(groups, groups[1:]):
            assert end == next_start
            assert 1 <= end - start <= accum
        assert all(end - start == accum for start, end in groups[:-1])

    def test_260_batches_accum2_gives_130_groups(self):
        groups = accumulation_groups(260, 2)
        assert len(groups) == 130
        assert groups[0] == (0, 2)
        assert groups[-1] == (258, 260)

    def test_zero_batches_gives_no_groups(self):
        assert accumulation_groups(0, 2) == []

    def test_invalid_accum_rejected(self):
        with pytest.raises(ValueError):
            accumulation_groups(4, 0)


class TestStepCounting:
    @pytest.mark.parametrize(
        "num_batches,accum",
        [
            (1, 1), (2, 1), (3, 1), (4, 1), (5, 1),
            (1, 2), (2, 2), (3, 2), (4, 2), (5, 2),
            (1, 3), (3, 3), (4, 3), (5, 3), (6, 3),
        ],
    )
    def test_steps_equal_ceil(self, num_batches, accum):
        model = TinyClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        loader = make_loader(num_batches, batch_size=1)
        stats = train_one_epoch(
            model, loader, optimizer, "cpu", accum=accum,
            limit_batches=num_batches, stage="finetune",
        )
        assert stats["batches"] == num_batches
        assert stats["optimizer_steps"] == math.ceil(num_batches / accum)

    def test_260_batches_accum2_steps_130(self):
        model = TinyClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        loader = make_loader(260, batch_size=1)
        stats = train_one_epoch(
            model, loader, optimizer, "cpu", accum=2, limit_batches=260,
            stage="finetune",
        )
        assert stats["batches"] == 260
        assert stats["optimizer_steps"] == 130

    def test_no_leftover_gradients_after_epoch(self):
        model = TinyClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        loader = make_loader(4, batch_size=1)
        train_one_epoch(
            model, loader, optimizer, "cpu", accum=2, limit_batches=4,
            stage="finetune",
        )
        assert all(p.grad is None for p in model.parameters())

    def test_truncated_epoch_steps_tail_group(self):
        # 7 batches available, truncated to 3 with accum 2 -> groups (2) + (1).
        model = TinyClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        loader = make_loader(7, batch_size=1)
        stats = train_one_epoch(
            model, loader, optimizer, "cpu", accum=2, limit_batches=3,
            stage="finetune",
        )
        assert stats["batches"] == 3
        assert stats["optimizer_steps"] == 2


class TestGradientNormalization:
    def _identical_models(self):
        torch.manual_seed(0)
        model = TinyClassifier()
        clone = TinyClassifier()
        clone.load_state_dict(model.state_dict())
        return model, clone

    def test_accumulated_gradient_equals_concatenated_group(self):
        """Unequal micro-batches 8+6 must equal one 14-sample update."""
        model, clone = self._identical_models()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        clone_optimizer = torch.optim.SGD(clone.parameters(), lr=0.05)

        # 14 samples as micro-batches 8 + 6 (one group, accum=2).
        loader = make_loader(14, batch_size=8)
        stats = train_one_epoch(
            model, loader, optimizer, "cpu", accum=2, limit_batches=2,
            stage="finetune",
        )
        assert stats["optimizer_steps"] == 1

        # The same 14 samples as a single batch (concatenated group).
        concat_loader = make_loader(14, batch_size=14)
        concat_stats = train_one_epoch(
            clone, concat_loader, clone_optimizer, "cpu", accum=1,
            limit_batches=1, stage="finetune",
        )
        assert concat_stats["optimizer_steps"] == 1

        for (name, value), (clone_name, clone_value) in zip(
            model.state_dict().items(), clone.state_dict().items()
        ):
            assert name == clone_name
            assert torch.allclose(value, clone_value, atol=1e-6), name

    def test_equal_micro_batches_match_concatenated(self):
        model, clone = self._identical_models()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        clone_optimizer = torch.optim.SGD(clone.parameters(), lr=0.05)
        loader = make_loader(16, batch_size=8)
        train_one_epoch(model, loader, optimizer, "cpu", accum=2, limit_batches=2,
                        stage="finetune")
        concat = make_loader(16, batch_size=16)
        train_one_epoch(clone, concat, clone_optimizer, "cpu", accum=1,
                        limit_batches=1, stage="finetune")
        for name, value in model.state_dict().items():
            assert torch.allclose(value, clone.state_dict()[name], atol=1e-6), name

    def test_epoch_loss_is_sample_weighted(self):
        """mean_loss must weight micro-batches by sample count, not by 1/k."""
        model = TinyClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # no updates
        loader = make_loader(22, batch_size=8)  # batches of 8, 8, 6
        stats = train_one_epoch(
            model, loader, optimizer, "cpu", accum=2, limit_batches=3,
            stage="finetune",
        )
        criterion = nn.CrossEntropyLoss()
        sizes = []
        losses = []
        model.eval()
        with torch.no_grad():
            for rgb, aux, quality, labels in loader:
                sizes.append(int(labels.shape[0]))
                losses.append(float(criterion(model(rgb, aux), labels)))
        assert sizes == [8, 8, 6]
        expected = sum(loss * n for loss, n in zip(losses, sizes)) / sum(sizes)
        assert stats["mean_loss"] == pytest.approx(expected, abs=1e-6)
        assert stats["samples"] == 22

    def test_unweighted_microbatch_mean_would_differ(self):
        """Guard: the fixture really distinguishes the two weightings."""
        model = TinyClassifier()
        criterion = nn.CrossEntropyLoss()
        loader = make_loader(22, batch_size=8)
        losses = []
        with torch.no_grad():
            for rgb, aux, quality, labels in loader:
                losses.append(float(criterion(model(rgb, aux), labels)))
        sizes = [8, 8, 6]
        weighted = sum(loss * n for loss, n in zip(losses, sizes)) / sum(sizes)
        unweighted = sum(losses) / len(losses)
        assert abs(weighted - unweighted) > 1e-4
