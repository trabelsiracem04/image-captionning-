import time

import torch
import torch.nn.utils as nn_utils

from src.training.losses import captioning_loss, doubly_stochastic_reg
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

        if self.cfg.training.fine_tune:
            self.model.encoder.train()
        else:
            self.model.encoder.eval()

        param_groups = [
            {
                "params": self.model.decoder.parameters(),
                "lr": self.cfg.training.lr_decoder,
            }
        ]
        if self.cfg.training.fine_tune:
            encoder_params = [
                p for p in self.model.encoder.parameters() if p.requires_grad
            ]
            param_groups.append(
                {"params": encoder_params, "lr": self.cfg.training.lr_cnn}
            )
        self.optimizer = torch.optim.AdamW(param_groups)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2
        )

        if self.cfg.training.fine_tune:
            trainable_enc = [p for p in self.model.encoder.parameters() if p.requires_grad]
            assert trainable_enc, "FINE-TUNE requested but encoder has no trainable params"
            assert len(self.optimizer.param_groups) == 2, (
                "FINE-TUNE requested but optimizer is not two-group"
            )
            enc_in_opt = any(
                p in set(self.optimizer.param_groups[1]["params"]) for p in trainable_enc
            )
            assert enc_in_opt, "FINE-TUNE requested but encoder params not in optimizer"
            self._log(
                "FINE-TUNE ACTIVE: %d encoder params trainable | encoder mode=%s"
                % (
                    len(trainable_enc),
                    "train" if self.model.encoder.training else "eval",
                )
            )

        self.best_bleu1 = -float("inf")
        self.start_epoch = 0
        self.batch_size = cfg.training.batch_size
        self.patience = getattr(cfg.training, "patience", 3)
        self.early_stopped = False

    def train_epoch(self):
        self.model.train()
        if self.cfg.training.fine_tune:
            self.model.encoder.train()
        else:
            self.model.encoder.eval()

        total_loss, n_batches = 0.0, 0
        t_start = time.time()
        for batch in self.train_loader:
            images = batch["images"].to(self.device)
            labels = batch["caption_ids"].to(self.device)

            logits, weights, _ = self.model(images, labels)
            ce = captioning_loss(logits, labels, pad_id=self.vocab.PAD)
            reg = doubly_stochastic_reg(weights, labels, pad_id=self.vocab.PAD)
            loss = ce + getattr(self.cfg.training, "attention_reg", 1.0) * reg

            self.optimizer.zero_grad()
            loss.backward()
            nn_utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.training.grad_clip
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
        load_optimizer = not self.cfg.training.fine_tune
        info = load_checkpoint(
            checkpoint_path,
            self.model,
            optimizer=self.optimizer if load_optimizer else None,
            device=self.device,
        )
        self.start_epoch = info["epoch"] + 1
        self.best_bleu1 = info["metrics"].get("best_bleu1", -float("inf"))
        self._log(f"resumed from {checkpoint_path} at epoch {info['epoch']}")