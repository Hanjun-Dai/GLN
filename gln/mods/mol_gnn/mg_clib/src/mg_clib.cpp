#include "mg_clib.h"
#include "config.h"
#include "mol_utils.h"
#include <random>
#include <algorithm>
#include <cstdlib>
#include <signal.h>

int Init(const int argc, const char **argv)
{
    cfg::LoadParams(argc, argv);
    return 0;
}


int PrepareBatchFeature(const int num_graphs,
                        const int *num_nodes,
                        const int *num_edges,
                        void** list_node_feats, 
                        void** list_edge_feats, 
                        Dtype* node_input, 
                        Dtype* edge_input)
{
    Dtype* ptr = node_input;
    for (int i = 0; i < num_graphs; ++i)
    {
        int* node_feats = static_cast<int*>(list_node_feats[i]);
        for (int j = 0; j < num_nodes[i]; ++j)
        {
            MolFeat::ParseAtomFeat(ptr, node_feats[j]);
            ptr += cfg::nodefeat_dim;
        }
    }

    ptr = edge_input;
    for (int i = 0; i < num_graphs; ++i)
    {
        int* edge_feats = static_cast<int*>(list_edge_feats[i]);
        for (int j = 0; j < num_edges[i] * 2; j += 2)
        {
            // two directions have the same feature
            MolFeat::ParseEdgeFeat(ptr, edge_feats[j / 2]);
            ptr += cfg::edgefeat_dim;
            MolFeat::ParseEdgeFeat(ptr, edge_feats[j / 2]);
            ptr += cfg::edgefeat_dim;
        }
    }

    return 0;
}


int PrepareIndices(const int num_graphs,
                   const int *num_nodes,
                   const int *num_edges,
                   void **list_of_edge_pairs, 
                   long long* edge_to_idx,
                   long long* edge_from_idx,
                   long long* g_idx)
{
    int offset = 0;
    int cur_edge = 0;    
    for (int i = 0; i < num_graphs; ++i)
    {
        int *edge_pairs = static_cast<int *>(list_of_edge_pairs[i]);
        for (int j = 0; j < num_edges[i] * 2; j += 2)
        {
            int x = offset + edge_pairs[j];
            int y = offset + edge_pairs[j + 1];
            edge_to_idx[cur_edge] = y;
            edge_from_idx[cur_edge] = x;
            cur_edge += 1;
            edge_to_idx[cur_edge] = x;
            edge_from_idx[cur_edge] = y;
            cur_edge += 1;
        }
        for (int j = 0; j < num_nodes[i]; ++j)
            g_idx[offset + j] = i;
        offset += num_nodes[i];
    }
    return 0;
}
