import ctypes
import numpy as np
import os
import sys
try:
    import torch
except:
    print('no torch loaded')

class _mg_lib(object):

    def __init__(self, sys_args):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.lib = ctypes.CDLL('%s/build/dll/libmolgnn.so' % dir_path)

        atom_file = '%s/default_atoms.txt' % dir_path
        for i in range(len(sys_args)):
            if sys_args[i] == '-f_atoms':
                atom_file = sys.argv[i + 1]
        atom_nums = []
        with open(atom_file, 'r') as f:
            for row in f:
                atom_nums.append(int(row.strip()))
                    
        self.lib.PrepareIndices.restype = ctypes.c_int
        self.lib.PrepareBatchFeature.restype = ctypes.c_int

        self.NUM_EDGE_FEATS = 7
        self.NUM_NODE_FEATS = len(atom_nums) + 23

        self.atom_idx_map = {}
        for i in range(len(atom_nums)):
            self.atom_idx_map[atom_nums[i]] = i

        args = 'this -num_atom_types %d -nodefeat_dim %d -edgefeat_dim %d' % (len(atom_nums), self.NUM_NODE_FEATS, self.NUM_EDGE_FEATS)
        args = args.split()
        if sys.version_info[0] > 2:
            args = [arg.encode() for arg in args]  # str -> bytes for each element in args
        
        arr = (ctypes.c_char_p * len(args))()
        arr[:] = args
        self.lib.Init(len(args), arr)

    def PrepareIndices(self, graph_list):
        edgepair_list = (ctypes.c_void_p * len(graph_list))()
        list_num_nodes = np.zeros((len(graph_list), ), dtype=np.int32)
        list_num_edges = np.zeros((len(graph_list), ), dtype=np.int32)        
        for i in range(len(graph_list)):
            if type(graph_list[i].edge_pairs) is ctypes.c_void_p:
                edgepair_list[i] = graph_list[i].edge_pairs
            elif type(graph_list[i].edge_pairs) is np.ndarray:
                edgepair_list[i] = ctypes.c_void_p(graph_list[i].edge_pairs.ctypes.data)
            else:
                raise NotImplementedError

            list_num_nodes[i] = graph_list[i].num_nodes
            list_num_edges[i] = graph_list[i].num_edges
        total_num_nodes = np.sum(list_num_nodes)
        total_num_edges = np.sum(list_num_edges)

        edge_to_idx = torch.LongTensor(total_num_edges * 2)
        edge_from_idx = torch.LongTensor(total_num_edges * 2)
        g_idx = torch.LongTensor(total_num_nodes)
        self.lib.PrepareIndices(len(graph_list), 
                                ctypes.c_void_p(list_num_nodes.ctypes.data),
                                ctypes.c_void_p(list_num_edges.ctypes.data),
                                ctypes.cast(edgepair_list, ctypes.c_void_p),
                                ctypes.c_void_p(edge_to_idx.numpy().ctypes.data),
                                ctypes.c_void_p(edge_from_idx.numpy().ctypes.data),
                                ctypes.c_void_p(g_idx.numpy().ctypes.data))
        return edge_to_idx, edge_from_idx, g_idx

    def PrepareBatchFeature(self, molgraph_list):
        n_graphs = len(molgraph_list)
        c_node_list = (ctypes.c_void_p * n_graphs)()
        c_edge_list = (ctypes.c_void_p * n_graphs)()
        list_num_nodes = np.zeros((n_graphs, ), dtype=np.int32)
        list_num_edges = np.zeros((n_graphs, ), dtype=np.int32)

        for i in range(n_graphs):
            mol = molgraph_list[i]
            c_node_list[i] = ctypes.c_void_p(mol.atom_feats.ctypes.data)
            c_edge_list[i] = ctypes.c_void_p(mol.bond_feats.ctypes.data)
            list_num_nodes[i] = mol.num_nodes
            list_num_edges[i] = mol.num_edges
    
        torch_node_feat = torch.zeros(np.sum(list_num_nodes), self.NUM_NODE_FEATS)
        torch_edge_feat = torch.zeros(np.sum(list_num_edges) * 2, self.NUM_EDGE_FEATS)

        node_feat = torch_node_feat.numpy()
        edge_feat = torch_edge_feat.numpy()

        self.lib.PrepareBatchFeature(n_graphs, 
                                     ctypes.c_void_p(list_num_nodes.ctypes.data),
                                     ctypes.c_void_p(list_num_edges.ctypes.data),
                                     ctypes.cast(c_node_list, ctypes.c_void_p),
                                     ctypes.cast(c_edge_list, ctypes.c_void_p),
                                     ctypes.c_void_p(node_feat.ctypes.data), 
                                     ctypes.c_void_p(edge_feat.ctypes.data))

        return torch_node_feat, torch_edge_feat


dll_path = '%s/build/dll/libmolgnn.so' % os.path.dirname(os.path.realpath(__file__))
if os.path.exists(dll_path):
    MGLIB = _mg_lib(sys.argv)
else:
    MGLIB = None
