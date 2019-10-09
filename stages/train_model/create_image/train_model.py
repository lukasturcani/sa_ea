import logging
from collections import Counter
import os
from os.path import join
import json
import rdkit.Chem.AllChem as rdkit
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.utils import shuffle
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    recall_score,
    precision_score,
)


logger = logging.getLogger(__name__)


def get_fingerprint(mol_block):
    info = {}
    fp = rdkit.GetMorganFingerprintAsBitVect(
        mol=rdkit.MolFromMolBlock(mol_block),
        radius=8,
        nBits=512,
        bitInfo=info,
    )
    fp = list(fp)
    for bit, activators in info.items():
        fp[bit] = len(activators)
    return fp


def get_dataset():
    molder = join(os.path.dirname(__file__), 'molder')

    with open(join(molder, 'database.json'), 'r') as f:
        database = json.load(f)

    with open(join(molder, 'Becky', 'opinions.json'), 'r') as f:
        opinions = json.load(f)

    fingerprints, labels = [], []
    for inchi, label in opinions.items():
        fingerprints.append(get_fingerprint(database[inchi]))
        labels.append(label)

    fingerprints, labels = np.array(fingerprints), np.array(labels)
    return shuffle(fingerprints, labels)


def main():
    np.random.seed(4)
    logging.basicConfig(level=logging.DEBUG)

    fingerprints, labels = get_dataset()
    logger.debug(f'Fingerprint shape is {fingerprints.shape}.')
    logger.debug(f'Collected labels are {Counter(labels)}.')

    clf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        class_weight='balanced',
    )

    scores = cross_validate(
        estimator=clf,
        X=fingerprints,
        y=labels,
        cv=5,
        scoring={
            'accuracy': make_scorer(accuracy_score),
            'precision_0': make_scorer(
                score_func=precision_score,
                pos_label=0,
                labels=[0],
            ),
            'recall_0': make_scorer(
                score_func=recall_score,
                pos_label=0,
                labels=[0],
            ),
            'precision_1': make_scorer(
                score_func=precision_score,
                pos_label=1,
                labels=[1],
            ),
            'recall_1': make_scorer(
                score_func=recall_score,
                pos_label=1,
                labels=[1],
            ),
        }
    )

    accuracy = scores['test_accuracy'].mean()
    p0 = scores['test_precision_0'].mean()
    r0 = scores['test_recall_0'].mean()
    p1 = scores['test_precision_1'].mean()
    r1 = scores['test_recall_1'].mean()

    print(f'accuracy\n{accuracy}', end='\n\n')
    print(f'precision (not sa)\n{p0}', end='\n\n')
    print(f'recall (not sa)\n{r0}', end='\n\n')
    print(f'precision (sa)\n{p1}', end='\n\n')
    print(f'recall (sa)\n{r1}', end='\n\n')

    clf = RandomForestClassifier(**clf.get_params())
    clf.fit(fingerprints, labels)

    with open('sa_model.pkl', 'wb') as f:
        pickle.dump(clf, f)


if __name__ == '__main__':
    main()
