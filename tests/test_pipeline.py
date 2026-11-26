import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from torch import nn

from ecg_mixmatch.cli import main
from ecg_mixmatch.mixmatch import augment_views, guess_labels, mix_batches


class MixMatchTests(unittest.TestCase):
    def test_pseudo_labels_average_probabilities_and_preserve_model_mode(self):
        model = nn.Identity()
        batch = torch.tensor([[[6.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
        expected = batch.softmax(-1).mean(dim=1).square()
        expected /= expected.sum(dim=-1, keepdim=True)
        torch.testing.assert_close(guess_labels(model, batch, 0.5), expected)
        self.assertTrue(model.training)

    def test_mixup_keeps_anchors(self):
        labeled = torch.zeros(4, 8)
        labels = torch.zeros(4, dtype=torch.long)
        views = torch.ones(4, 2, 8)
        pseudo = torch.tensor([[0.0, 1.0, 0.0]]).expand(4, 3)
        mixed, targets = mix_batches(
            labeled, labels, views, pseudo, alpha=0.75, rng=np.random.default_rng(2)
        )
        self.assertTrue((mixed[:4] < 0.5).all())
        self.assertTrue((mixed[4:] > 0.5).all())
        torch.testing.assert_close(mixed[:, 0], targets[:, 1])

    def test_augmentations_do_not_mutate_signals(self):
        signals = torch.ones(4, 16)
        first = augment_views(signals, 2)
        second = augment_views(signals, 2)
        self.assertEqual(first.shape, (4, 2, 16))
        self.assertFalse(torch.equal(first, second))
        torch.testing.assert_close(signals, torch.ones_like(signals))


class WorkflowTests(unittest.TestCase):
    def test_train_and_evaluate(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            archive, output = root / "ecg.npz", root / "run"
            rng = np.random.default_rng(7)
            np.savez(
                archive,
                ecgstrain=rng.normal(size=(36, 65, 1)),
                labelstrain=np.tile(["N", "A", "O", "~"], 9),
                ecgsval=rng.normal(size=(9, 1, 65)),
                labelsval=np.tile(np.eye(3), (3, 1)),
                ecgstest=rng.normal(size=(8, 65)),
                labelstest=np.tile([0, 1], 4),
            )
            shared = ["--data", str(archive), "--device", "cpu", "--cpu-threads", "1"]
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                main([
                    "train", *shared, "--output", str(output), "--iterations", "4",
                    "--eval-interval", "3", "--batch-size", "4", "--labeled-ratio", "0.4",
                    "--base-channels", "8", "--gru-hidden", "8",
                ])
            self.assertEqual(json.loads(stdout.getvalue())["split_sizes"]["train"], 27)
            history = json.loads((output / "history.json").read_text())
            self.assertEqual([entry["step"] for entry in history["validation"]], [3, 4])

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                main(["evaluate", *shared, "--checkpoint", str(output / "best_model.pth")])
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["class_names"], ["N", "A", "O"])
            self.assertEqual(np.asarray(result["confusion_matrix"]).shape, (3, 3))
            self.assertEqual(result["per_class"]["O"]["support"], 0)
            self.assertEqual(result["samples"], 8)


if __name__ == "__main__":
    unittest.main()
