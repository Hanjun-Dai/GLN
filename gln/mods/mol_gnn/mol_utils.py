import ctypes
import numpy as np
import os
import sys
from tqdm import tqdm
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import struct
import pickle as cp
from gln.mods.mol_gnn.mg_clib.mg_lib import MGLIB


def get_atom_feat(atom, sanitized):
    # getIsAromatic
    feat = int(atom.GetIsAromatic())
    feat <<= 4
    # getImplicitValence
    v = atom.GetImplicitValence()
    if 0 <= v <= 5:
        feat |= v
    else:
        feat |= 6
    feat <<= 4
    # getTotalNumHs
    if sanitized:
        h = atom.GetTotalNumHs()
        if h <= 4:
            feat |= h
        else:
            feat |= 4
    else:
        feat |= 5
    feat <<= 4
    # getDegree
    feat |= atom.GetDegree()    
    feat <<= 8
    x = atom.GetAtomicNum()
    if x in MGLIB.atom_idx_map:
        feat |= MGLIB.atom_idx_map[x]
    else:
        feat |= len(MGLIB.atom_idx_map)
    assert feat >= 0
    return feat


def get_bond_feat(bond, sanitized):
    bt = bond.GetBondType()
    t = 0
    if bt == rdkit.Chem.rdchem.BondType.SINGLE:
        t = 0
    elif bt == rdkit.Chem.rdchem.BondType.DOUBLE:
        t = 1
    elif bt == rdkit.Chem.rdchem.BondType.TRIPLE:
        t = 2
    elif bt == rdkit.Chem.rdchem.BondType.AROMATIC:
        t = 3
    feat = 2
    if sanitized:
        feat = 1 if bond.GetOwningMol().GetRingInfo().NumBondRings(bond.GetIdx()) > 0 else 0
    feat <<= 8
    feat |= int(bond.GetIsConjugated())
    feat <<= 8
    feat |= t
    assert feat >= 0
    return feat


class MolGraph(object):

    def __init__(self, name, sanitized, *, mol=None, num_nodes=None, num_edges=None, atom_feats=None, bond_feats=None, edge_pairs=None):
        self.name = name
        self.sanitized = sanitized
        if num_nodes is None:
            assert mol is not None
            self.num_nodes = mol.GetNumAtoms()
            self.num_edges = mol.GetNumBonds()

            self.atom_feats = np.zeros((self.num_nodes, ), dtype=np.int32)
            for i, atom in enumerate(mol.GetAtoms()):
                self.atom_feats[i] = get_atom_feat(atom, self.sanitized)
            
            self.bond_feats = np.zeros((self.num_edges, ), dtype=np.int32)
            self.edge_pairs = np.zeros((self.num_edges * 2, ), dtype=np.int32)
        
            for i, bond in enumerate(mol.GetBonds()):
                self.bond_feats[i] = get_bond_feat(bond, self.sanitized)
                x = bond.GetBeginAtomIdx()
                y = bond.GetEndAtomIdx()
                self.edge_pairs[i * 2] = x
                self.edge_pairs[i * 2 + 1] = y
        else:
            self.num_nodes = num_nodes
            self.num_edges = num_edges

            self.atom_feats = np.array(atom_feats, dtype=np.int32) if atom_feats is not None else None
            self.bond_feats = np.array(bond_feats, dtype=np.int32) if bond_feats is not None else None
            self.edge_pairs = np.array(edge_pairs, dtype=np.int32) if edge_pairs is not None else None
        self.fingerprints = None
        self.fp_info = None


class _MolHolder(object):

    def __init__(self, sanitized):
        self.sanitized = sanitized
        self.dict_molgraph = {}
        self.fp_degree = 0
        self.fp_info = False
        self.null_graphs = set()

    def set_fp_degree(self, degree, fp_info=False):
        self.fp_degree = degree
        self.fp_info = fp_info

    def _get_inv(self, m):
        if self.sanitized:
            return None
        feats = []
        for a in m.GetAtoms():
            f = (a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge())
            f = ctypes.c_uint32(hash(f)).value
            feats.append(f)
        return feats

    def new_mol(self, name):
        if self.sanitized:
            mol = Chem.MolFromSmiles(name)
        else:
            mol = Chem.MolFromSmarts(name)
        if mol is None:            
            return None
        else:
            mg = MolGraph(name, self.sanitized, mol=mol)
            if self.fp_degree > 0:
                bi = {} if self.fp_info else None
                feat = AllChem.GetMorganFingerprint(mol, self.fp_degree, bitInfo=bi, invariants=self._get_inv(mol))
                on_bits = list(feat.GetNonzeroElements().keys())
                mg.fingerprints = on_bits
                mg.fp_info = bi
            return mg

    def get_mol_graph(self, name):
        if name is None or len(name.strip()) == 0:
            return None
        if name in self.null_graphs:
            return None
        if not name in self.dict_molgraph:
            mg = self.new_mol(name)
            if mg is None:
                self.null_graphs.add(name)
                return None
            else:
                self.dict_molgraph[name] = mg
        return self.dict_molgraph[name]

    def clear(self):
        self.dict_molgraph = {}

    def save_dump(self, prefix):
        with open(prefix + '.names', 'w') as f:
            for key in self.dict_molgraph:
                f.write('%s\n' % key)
        
        with open(prefix + '.bin', 'wb') as f:
            n_graphs = len(self.dict_molgraph)
            # write total number of mols
            f.write(struct.pack('=i', n_graphs))
            # save all the size info
            list_num_nodes = [None] * n_graphs
            list_num_edges = [None] * n_graphs
            for i, key in enumerate(self.dict_molgraph):
                mol = self.dict_molgraph[key]
                list_num_nodes[i] = mol.num_nodes
                list_num_edges[i] = mol.num_edges
            
            f.write(struct.pack('=%di' % n_graphs, *list_num_nodes))
            f.write(struct.pack('=%di' % n_graphs, *list_num_edges))

            for key in tqdm(self.dict_molgraph):
                mol = self.dict_molgraph[key]
                f.write(struct.pack('=%di' % mol.num_nodes, *(mol.atom_feats.tolist())))
                f.write(struct.pack('=%di' % mol.num_edges, *(mol.bond_feats.tolist())))
                f.write(struct.pack('=%di' % (mol.num_edges * 2), *(mol.edge_pairs.tolist())))

        if self.fp_degree > 0:
            if self.fp_info:
                with open(prefix + '.fp%d_info' % self.fp_degree, 'wb') as f:
                    for key in self.dict_molgraph:
                        mol = self.dict_molgraph[key]
                        assert mol.fp_info is not None
                        cp.dump(mol.fp_info, f, cp.HIGHEST_PROTOCOL)
            else:
                with open(prefix + '.fp%d' % self.fp_degree, 'wb') as f:
                    num_fps = [None] * n_graphs
                    for i, key in enumerate(self.dict_molgraph):
                        mol = self.dict_molgraph[key]
                        num_fps[i] = len(mol.fingerprints)
                    f.write(struct.pack('=%di' % n_graphs, *num_fps))
                    for i, key in enumerate(self.dict_molgraph):
                        mol = self.dict_molgraph[key]
                        fp = mol.fingerprints
                        f.write(struct.pack('=%dI' % len(fp), *fp))

        print('%d molecules saved' % n_graphs)
        print('total # nodes', np.sum(list_num_nodes))
        print('total # edges', np.sum(list_num_edges))

    def remove_dump(self, prefix):
        print('mol_holder unloading', prefix)
        names = []
        with open(prefix + '.names', 'r') as f:
            for row in f:
                names.append(row.strip())
        [self.dict_molgraph.pop(x, None) for x in names]

    def load_dump(self, prefix, additive=False, load_feats=True, load_fp=True):
        print('mol_holder loading', prefix)
        if not additive:
            self.dict_molgraph = {}
        names = []
        with open(prefix + '.names', 'r') as f:
            for row in f:
                names.append(row.strip())
                self.dict_molgraph[names[-1]] = MolGraph(names[-1], self.sanitized, num_nodes=-1, num_edges=-1)
        if load_feats:
            print('loading binary features')
            with open(prefix + '.bin', 'rb') as f:
                n_graphs = struct.unpack('=i', f.read(4))[0]            
                assert n_graphs == len(names)
                list_num_nodes = struct.unpack('=%di' % n_graphs, f.read(4 * n_graphs))
                list_num_edges = struct.unpack('=%di' % n_graphs, f.read(4 * n_graphs))

                for i in tqdm(range(n_graphs)):
                    mol = self.dict_molgraph[names[i]]
                    mol.num_nodes = n = list_num_nodes[i]
                    mol.num_edges = m = list_num_edges[i]

                    mol.atom_feats = np.array(struct.unpack('=%di' % n, f.read(4 * n)), dtype=np.int32)
                    mol.bond_feats = np.array(struct.unpack('=%di' % m, f.read(4 * m)), dtype=np.int32)
                    mol.edge_pairs = np.array(struct.unpack('=%di' % (2 * m), f.read(4 * 2 * m)), dtype=np.int32)

        if self.fp_degree > 0 and load_fp:
            print('loading fingerprints')
            if self.fp_info:
                with open(prefix + '.fp%d_info' % self.fp_degree, 'rb') as f:
                    for name in tqdm(names):
                        d = cp.load(f)
                        mol = self.dict_molgraph[name]
                        mol.fp_info = d
                        mol.fingerprints = list(d.keys())
            else:
                n_graphs = len(names)
                with open(prefix + '.fp%d' % self.fp_degree, 'rb') as f:
                    num_fps = struct.unpack('=%di' % n_graphs, f.read(4 * n_graphs))
                    for i, key in tqdm(enumerate(names)):
                        mol = self.dict_molgraph[key]
                        mol.fingerprints = struct.unpack('=%dI' % num_fps[i], f.read(4 * num_fps[i]))                        

            print('done with fp loading')

SmartsMols = _MolHolder(sanitized=False)
SmilesMols = _MolHolder(sanitized=True)
