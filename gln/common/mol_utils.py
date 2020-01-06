from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import rdkit
from rdkit import Chem

def cano_smiles(smiles):
    try:
        tmp = Chem.MolFromSmiles(smiles)
        if tmp is None:
            return None, smiles        
        tmp = Chem.RemoveHs(tmp)
        if tmp is None:
            return None, smiles
        [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
        return tmp, Chem.MolToSmiles(tmp)            
    except:
        return None, smiles


def cano_smarts(smarts):
    tmp = Chem.MolFromSmarts(smarts)
    if tmp is None:        
        return None, smarts
    [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    cano = Chem.MolToSmarts(tmp)
    if '[[se]]' in cano:  # strange parse error
        cano = smarts
    return tmp, cano


def smarts_has_useless_parentheses(smarts):
    if len(smarts) == 0:
        return False
    if smarts[0] != '(' or smarts[-1] != ')':
        return False
    cnt = 1
    for i in range(1, len(smarts)):
        if smarts[i] == '(':
            cnt += 1
        if smarts[i] == ')':
            cnt -= 1
        if cnt == 0:
            if i + 1 != len(smarts):
                return False
    return True
