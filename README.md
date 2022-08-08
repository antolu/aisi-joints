# aisi-joints
Code relating to development of computer vision model for detecting defective rail joints on the Swiss railways, hosted at SBB CFF FFS in Bern, Switzerland as oart of SBB Infrastructure.

## Requirements

Python 3.8  
Tensorflow 2.6.0  
[Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) installed according to the [official instructions](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html).  
Numpy >= 1.20
All requirements are listed in [setup.cfg](setup.cfg) file.

**See below installation instructions for formal requirements**

## Installation

The package is intended to be used as a module (`python -m aisi_joints...`), and should thus be installed to python path.
A [setup.py](setup.py) is provided for easy installation.
The Tensorflow Object Detection API requires Tensorflow 2.5.0, while the image classification approach uses Tensorflow >= 2.8.0. Therefore dynamic dependency management is necessary, and in principle it is not possible to run TFOD in the same virtual environment as `img_cls` submodule.

Each submodule has its own set of dependencies, and can be installed by specifying an "extra" when installing using pip.

For instance, running
```shell
pip install .[img_cls]
```

installs dependencies for [img_cls](aisi_joints/img_cls) submodule for the Tensorflow image classification approach, while

```shell
pip install .[detr]
```

installs dependencies for the [detr](aisi_joints/detr) submodule for DE:TR object detection.

The following optional dependency "extra"s are available:

* detr
* fiftyone
* img_cls
* tfod
* self_supervised

### Tensorflow Object Detection API

Note that the `object_detection` package for tensorflow object detection is not installed automatically by pip, and needs to be installed automatically according to the [official instructions](https://github.com/tensorflow/models/tree/master/research/object_detection) installed according to the [official instructions](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html).
A [convenience script](scripts/install_object_detection.py) exists to install `object_detection` automatically. Note that this script does not work on Apple Silicon based systems since it uses `tensorflow-macos` as dependency instead of `tensorflow`.

### Pycocotools

Pycocotools needs to be patched (see this issue) in order to allow per-category detection metrics.
A [convenience script](scripts/update_pycocotools.py) exists to patch the `pycocotools` and `object_detection` code automatically.

## How to use

### Data preprocessing

To process raw data from RCM API, use the `aisi_joints.data` module. First the .csv files provided by RCM API needs to be cleaned, and unlabeled data filtered out. The script matches label to box and image using the eventId UUID. This is done using

```shell
python -m aisi_joints.data.import_rcm_api -l /path/to/csv/with/labels.csv -b /path/to/csv/with/boxes.csv -i /path/to/directory/with/images
```
which generates the file `output.csv` containing only the labeled data, as well as a labelmap named `output_labelmap.pbtxt`

The next step consists of splitting the dataset into train/validation(/test) splits, using the following:

```shell
python -m aisi_joints.data.partition_dataset [ratio] /path/to/preprocessed.csv
```

Where ratio can be `80/20` for an 80/20 split between training and validation set, or `80/10/10` for 80/10/10 split between training, validation and test set. The partitioning is done evenly between images recorded using DFZ and gDFZ using the `platformId` column. By default the script outputs the `output_split.csv` file.

Lastly the split dataset needs to be converted into tfrecord format for the Tensorflow Object Detection API.

```shell
python -m aisi_joints.data.generate_tfrecord -l /path/to/labelmap.pbtxt -i /path/to/output_split.csv
```
Which outputs one tfrecord file per split, e.g. `train.tfrecord`,`validation.tfrecord`,`test.tfrecord`.
