from argparse import ArgumentParser
from .fiftyone import main

parser = ArgumentParser()

parser.add_argument('csv_path',
                    help='Path to file containing csv')
args = parser.parse_args()

main(args)
