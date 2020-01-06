from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import rdkit
from rdkit import Chem

def canonicalize(smiles):
    try:        
        tmp = Chem.MolFromSmiles(smiles)
    except:
        print('no mol')
        return smiles
    if tmp is None:
        return smiles
    tmp = Chem.RemoveHs(tmp)
    [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    return Chem.MolToSmiles(tmp)

def get_weighted_f1(seq_pred, seq_gnd):
    if seq_pred is None or seq_gnd is None:
        return 0.0
    pred = set(seq_pred.split('.'))
    gnd = set(seq_gnd.split('.'))

    t = pred.intersection(gnd)
    w = len(t) / float(len(gnd))
    precision = len(t) / float(len(pred))
    recall = len(t) / float(len(gnd))
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall * w / (precision + recall)

def get_score(pred, gnd, score_type):
    x = canonicalize(gnd)
    y = canonicalize(pred)
    if score_type == 'mix_f1':
        f1 = get_weighted_f1(y, x)
        score = 0.75 * f1 + 0.25 * (x == y)
    elif score_type == 'acc':
        score = 1.0 * (x == y)
    else:
        raise NotImplementedError
    return score
