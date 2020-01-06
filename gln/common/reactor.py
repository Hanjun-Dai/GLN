from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import rdkit
from rdkit import Chem
from gln.common.cmd_args import rdchiralReaction, rdchiralReactants, rdchiralRun

class _Reactor(object):

    def __init__(self):
        self.rxn_cooked = {}
        self.src_cooked = {}
        self.cached_results = {}

    def get_rxn(self, rxn):
        p, a, r = rxn.split('>')
        if '.' in p:  # we assume the product has only one molecule
            if p[0] != '(':
                p = '('+p+')'
        rxn = '>'.join((p, a, r))
        if not rxn in self.rxn_cooked:
            try:
                t = rdchiralReaction(rxn)
            except:
                t = None
            self.rxn_cooked[rxn] = t
        return self.rxn_cooked[rxn]

    def get_src(self, smiles):
        if not smiles in self.src_cooked:
            self.src_cooked[smiles] = rdchiralReactants(smiles)
        return self.src_cooked[smiles]

    def run_reaction(self, src, template):
        key = (src, template)
        if key in self.cached_results:
            return self.cached_results[key]
        rxn = self.get_rxn(template)
        src = self.get_src(src)
        if rxn is None or src is None:
            return None
        try:
            outcomes = rdchiralRun(rxn, src)
            self.cached_results[key] = outcomes
        except:
            self.cached_results[key] = None
        return self.cached_results[key]


Reactor = _Reactor()