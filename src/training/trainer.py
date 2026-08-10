import time

import torch
import torch.nn.utils as nn_utils

from src.training.losses import captioning_loss
from src.training.validate import validate_captions
from src.utils.checkpoints import checkpoint_paths, save_checkpoint, load_checkpoint


class Trainer:
    """Stage-1 trainer: CNN encoder frozen, decoder/attention trained."""

    def __init__(
        self,
        cfg,
        model,
        vocab,
        device,
        train_loader,
        val_loader=None,
        references_by_id=None,
        checkpoints_dir=None,
        logger=None,
    ):
        self.cfg = cfg
        self.model = model
        self.vocab = vocab
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.references_by_id = references_by_id or {}
        self.checkpoints_dir = checkpoints_dir
        self.logger = logger

        self.model.encoder.eval()  # frozen backbone, BN stays frozen during stage 1
        self.optimizer = torch.optim.AdamW(
            self.model.decoder.parameters(), lr=cfg.training.lr_decoder
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2
        )

        self.best_bleu1 = -float("inf")
        self.start_epoch = 0
        self.batch_size = cfg.training.batch_size
        self.patience = getattr(cfg.training, "patience", 3)
        self.early_stopped = False

    def train_epoch(self):
        self.model.train()
        self.model.encoder.eval()

        total_loss, n_batches = 0.0, 0
        t_start = time.time()
        for batch in self.train_loader:
            images = batch["images"].to(self.device)
            labels = batch["caption_ids"].to(self.device)

            logits, _, _ = self.model(images, labels)
            loss = captioning_loss(logits, labels, pad_id=self.vocab.PAD)

            self.optimizer.zero_grad()
            loss.backward()
            nn_utils.clip_grad_norm_(
                self.model.decoder.parameters(), self.cfg.training.grad_clip
            )
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1), time.time() - t_start

    def validate(self, max_samples=None):
        return validate_captions(
            self.model,
            self.val_loader,
            self.vocab,
            self.device,
            self.references_by_id,
            max_length=self.cfg.inference.max_length,
            max_samples=max_samples,
        )

    def _log(self, message, level="info"):
        if self.logger is None:
            print(message)
            return
        getattr(self.logger, level)(message)

    def train(self, epochs):
        no_improve = 0
        epoch = self.start_epoch - 1
        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            loss, seconds = self.train_epoch()

            line = f"epoch {epoch + 1}/{self.start_epoch + epochs}  loss {loss:.4f}  ({seconds:.0f}s)"
            improved = False
            if self.val_loader is not None:
                metrics, samples = self.validate()
                bleu1 = metrics[1]
                improved = bleu1 > self.best_bleu1
                if improved:
                    self.best_bleu1 = bleu1
                    no_improve = 0
                else:
                    no_improve += 1
                line += (
                    f"  |  val BLEU-1 {bleu1:.4f}  BLEU-4 {metrics[4]:.4f}"
                    f"  {'(best)' if improved else ''}"
                )
                self.scheduler.step(bleu1)
                if self.logger is not None and samples:
                    self.logger.info("  sample: %s", samples[0]["caption"])
            self._log(line)

            if self.checkpoints_dir is not None:
                self._save_last(epoch)
                if improved:
                    self._save_best(epoch, self.best_bleu1)

            if self.val_loader is not None and no_improve >= self.patience:
                self.early_stopped = True
                self._log(
                    f"early stopping: no val BLEU-1 improvement for {self.patience} epochs"
                )
                break

        self.final_epoch = epoch
        return self.best_bleu1

    def _save_last(self, epoch):
        path = checkpoint_paths(self.checkpoints_dir)[1]
        save_checkpoint(
            path,
            self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            stage=1,
            metrics={"best_bleu1": self.best_bleu1},
        )

    def _save_best(self, epoch, bleu1):
        path = checkpoint_paths(self.checkpoints_dir)[0]
        save_checkpoint(
            path,
            self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            stage=1,
            metrics={"best_bleu1": bleu1},
        )

    def resume(self, checkpoint_path):
        info = load_checkpoint(
            checkpoint_path,
            self.model,
            optimizer=self.optimizer,
            device=self.device,
        )
        self.start_epoch = info["epoch"] + 1
        self.best_bleu1 = info["metrics"].get("best_bleu1", -float("inf"))
        self._log(f"resumed from {checkpoint_path} at epoch {info['epoch']}")