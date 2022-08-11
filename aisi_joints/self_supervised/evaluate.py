import logging
import os
from argparse import ArgumentParser
from pprint import pformat
from typing import List, Optional

import pandas as pd
import torch
from self_supervised.utils import get_class_dataset
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from self_supervised import LinearClassifierMethod
from .._utils.logging import setup_logger
from .._utils.utils import time_execution, get_latest
from ..constants import CLASS_OK, CLASS_DEFECT

log = logging.getLogger(__name__)


def evaluate(df: pd.DataFrame, model: LinearClassifierMethod) -> pd.DataFrame:
    """
    Calculate predictions from input data
    Print results to terminal.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalize_means = [0.28513786, 0.28513786, 0.28513786]
    normalize_stds = [0.21466085, 0.21466085, 0.21466085]

    transforms_ = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=normalize_means,
                std=normalize_stds,
            ),
        ]
    )

    dataset = get_class_dataset('aisi')

    dataloader = DataLoader(
        # JointDataset.from_df(df, None, False, 224, 224, transforms_),
        dataset.configure_test(),
        batch_size=model.hparams.batch_size,
        num_workers=model.hparams.num_data_workers,
    )

    # dataloader = model.val_dataloader()

    model.eval()
    model = model.to(device)

    model_outputs = []
    all_labels = []
    with time_execution() as t:
        for i, batch in enumerate(dataloader):
            input_, labels = batch
            input_, labels = input_.to(device), labels.to(device)
            logits = model(input_)

            logits = logits.detach().to('cpu')

            model_outputs.append(logits)
            all_labels.append(labels)

    logits = torch.cat(model_outputs)
    labels = torch.cat(all_labels).cpu().numpy()

    predictions = torch.nn.functional.softmax(logits)

    log.info(
        f'Done. Took {t.duration * 1000 / len(labels)} ms ' f'per sample.'
    )
    log.info('Calculating evaluation metrics.')

    scores, pred_labels = (o.numpy() for o in torch.max(predictions, dim=1))

    report = classification_report(
        labels,
        pred_labels,
        target_names=[CLASS_OK, CLASS_DEFECT],
        output_dict=True,
    )
    cf = confusion_matrix(labels, pred_labels)

    log.info(
        ('=' * 10)
        + 'CLASSIFICATION REPORT'
        + ('=' * 10)
        + '\n'
        + pformat(report)
        + '\n'
    )
    log.info(('=' * 10) + 'CONFUSION MATRIX' + ('=' * 10) + '\n' + pformat(cf))

    if df is None:
        return

    df = df.assign(detected_class=pred_labels)

    class_map = {CLASS_OK: 0, CLASS_DEFECT: 1}
    df['detected_class'] = df['detected_class'].map(
        {v: k for k, v in class_map.items()}
    )
    df['detection_score'] = scores
    df['num_detections'] = 1
    df['detected_x0'] = df['x0']
    df['detected_x1'] = df['x1']
    df['detected_y0'] = df['y0']
    df['detected_y1'] = df['y1']

    return df


def main(argv: Optional[List[str]] = None):
    parser = ArgumentParser()

    parser.add_argument(
        '-d',
        '--dataset',
        required=True,
        help='Path to .csv containing dataset, '
        'or path to directory containing images.',
    )
    parser.add_argument(
        '-m',
        '--model',
        dest='model',
        default='checkpoints',
        help='Path to directory containing save model ' '(state dict).',
    )
    parser.add_argument(
        '-s',
        '--split',
        choices=['train', 'validation', 'test'],
        default=None,
        help='Specific split to evaluate on.',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default=None,
        help='Output .csv with predictions.',
    )

    args = parser.parse_args(argv)

    setup_logger()

    os.environ['DATA_PATH'] = args.dataset
    df = pd.read_csv(args.dataset)

    if args.split is not None and 'split' in df.columns:
        df = df[df['split'] == args.split]
        log.info(f'Evaluating on {args.split} split with {len(df)} samples.')
    else:
        log.info('Evaluating on entire dataset.')

    if os.path.isfile(args.model):
        classifier_checkpoint = args.model
    elif os.path.isdir(args.model):
        classifier_checkpoint = get_latest(
            args.model,
            lambda o: o.startswith('model-classifier') and o.endswith('.ckpt'),
        )
    else:
        raise FileNotFoundError(
            'Could not find checkpoint file at ' + args.model
        )

    log.info(f'Loading model from {classifier_checkpoint}.')
    # need to load base weights and classifier weights separately

    model = LinearClassifierMethod.from_trained_checkpoint(
        classifier_checkpoint
    )
    log.info('Model loaded.')

    df = evaluate(df, model)

    if args.output is not None:
        log.info(f'Writing predictions to {args.output}')
        df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
