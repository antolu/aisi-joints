from argparse import ArgumentParser

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from transformers import DetrFeatureExtractor

from ._data import CocoDetection
from ._detr import Detr


def train(img_folder: str):
    feature_extractor = DetrFeatureExtractor.from_pretrained(
        "facebook/detr-resnet-50")

    train_dataset = CocoDetection(img_folder, 'train',
                                  feature_extractor=feature_extractor)
    val_dataset = CocoDetection(img_folder, 'validation',
                                feature_extractor=feature_extractor)

    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        encoding = feature_extractor.pad_and_create_pixel_mask(
            pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn,
                                  batch_size=32, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn,
                                batch_size=1, shuffle=False, num_workers=4)

    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, num_classes=2)
    trainer = Trainer(gpus=1, max_steps=10000, gradient_clip_val=1.0)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('data_dir', help='Path to COCO dataset folder.')

    args = parser.parse_args()

    train(args.data_dir)
