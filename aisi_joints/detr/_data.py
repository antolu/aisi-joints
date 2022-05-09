import torchvision
import os


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, dataset_folder: str, split_name: str,
                 feature_extractor):
        ann_file = os.path.join(dataset_folder, 'annotations',
                                f'{split_name}.json')
        img_folder = os.path.join(dataset_folder, split_name)

        super().__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super().__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target,
                                          return_tensors="pt")
        pixel_values = encoding[
            "pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target
