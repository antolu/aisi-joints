import pytorch_lightning as pl
import torch
from transformers import DetrForObjectDetection, \
    DetrFeatureExtractor


class Detr(pl.LightningModule):
    def __init__(self, lr: float, lr_backbone: float, weight_decay: float,
                 num_classes: int):
        super().__init__()

        normalize_means = [0.28513786, 0.28513786, 0.28513786]
        normalize_stds = [0.21466085, 0.21466085, 0.21466085]
        # replace COCO classification head with custom head
        self.feature_extractor = DetrFeatureExtractor.from_pretrained(
            "facebook/detr-resnet-101",
            mean=normalize_means, std=normalize_stds)
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-101",
            num_labels=num_classes,
            ignore_mismatched_sizes=True)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in
                  batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask,
                             labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss, batch_size=len(batch['labels']))
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item(), batch_size=len(batch['labels']))

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, batch_size=len(batch['labels']))
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item(),
                     batch_size=len(batch['labels']))

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if
                        "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if
                           "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                      weight_decay=self.weight_decay)

        return optimizer
