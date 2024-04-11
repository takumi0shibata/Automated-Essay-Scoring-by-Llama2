import pandas as pd
import numpy as np
import random
import os
import torch
from sklearn.metrics import cohen_kappa_score
from typing import List, Tuple, Dict, Union


def get_min_max_scores():
    return {
        1: {'score': (2, 12), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
            'sentence_fluency': (1, 6), 'conventions': (1, 6)},
        2: {'score': (1, 6), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
            'sentence_fluency': (1, 6), 'conventions': (1, 6)},
        3: {'score': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        4: {'score': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        5: {'score': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        6: {'score': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        7: {'score': (0, 30), 'content': (0, 6), 'organization': (0, 6), 'conventions': (0, 6),
        'style': (0, 6)
        },
        8: {'score': (0, 60), 'content': (2, 12), 'organization': (2, 12), 'word_choice': (2, 12),
            'sentence_fluency': (2, 12), 'conventions': (2, 12),
            'voice': (2, 12)
            }}


def load_data(data_path: str, attribute: str = 'score') -> Dict:
    """
    Load data from the given path.
    Args:
        data_path: Path to the data.
    Returns:
        dict: ASAP Dataset
    """
    data = {}
    for file in ['train', 'dev', 'test']:
        feature = []
        label = []
        essay_id = []
        essay_set = []
        try:
            read_data = pd.read_pickle(data_path + file + '.pkl')
        except:
            read_data = pd.read_pickle(data_path + file + '.pk')
        for i in range(len(read_data)):
            feature.append(read_data[i]['content_text'])
            label.append(int(read_data[i][attribute]))
            essay_id.append(int(read_data[i]['essay_id']))
            essay_set.append(int(read_data[i]['prompt_id']))
        data[file] = {'feature': feature, 'label': label, 'essay_id': essay_id, 'essay_set': essay_set}

    return data


def normalize_scores(
    y: np.ndarray,
    essay_set: np.ndarray,
    attribute_name: str
) -> np.ndarray:
    """
    Normalize scores based on the min and max scores for each unique prompt_id in essay_set.
    Args:
        y: Scores to normalize.
        essay_set: Array of essay_set (prompt_id) for each score.
        attribute_name: The attribute name to filter the min and max scores.
    Returns:
        np.ndarray: Normalized scores.
    """
    min_max_scores = get_min_max_scores()
    normalized_scores = np.zeros_like(y, dtype=float)
    for unique_prompt_id in np.unique(essay_set):
        minscore, maxscore = min_max_scores[unique_prompt_id][attribute_name]
        mask = (essay_set == unique_prompt_id)
        normalized_scores[mask] = (y[mask] - minscore) / (maxscore - minscore)
    return normalized_scores


def calc_kappa(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    prompt_id: int,
    attribute: str,
    weights: str = 'quadratic'
) -> float:
    """
    Calculate the cohen kappa score.
    Args:
        y_true: True labels
        y_pred: Predicted labels
        prompt_id: Prompt ID
        attribute: Attribute name
        weights: Type of weights to use for the kappa calculation
            quadratic: Quadratic weights
            linear: Linear weights
    Returns:
        float: Quadratic weighted kappa
    """

    minscore, maxscore = get_min_max_scores()[prompt_id][attribute]

    y_true = np.round((maxscore - minscore) * np.array(y_true) + minscore)
    y_pred = np.round((maxscore - minscore) * np.array(y_pred) + minscore).flatten()
    
    return cohen_kappa_score(y_true, y_pred, weights=weights, labels=[i for i in range(minscore, maxscore+1)])


def set_seed(seed: int) -> None:
    # fix random seed
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)