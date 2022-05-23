"""
This module contains code related to the Image Classification approach of the
joints model.
"""

install_requires = [
    'tensorflow>=2.8.0',
    'object_detection',  # only for visualisation in EvaluateImages
    'scikit-learn',  # for classification report and confusion matrix
    'pandas'
]
