# aisi-joints
AISI rail joint fault detection


## Requirements

Python 3.8  
Tensorflow 2.6.0  
[Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) installed according to the [official instructions](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html).  
Numpy >= 1.20

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
