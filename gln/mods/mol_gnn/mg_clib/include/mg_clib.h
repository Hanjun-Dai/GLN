#ifndef MG_CLIB_H
#define MG_CLIB_H

#include "config.h"

extern "C" int Init(const int argc, const char **argv);

extern "C" int PrepareBatchFeature(const int num_graphs,
                      	           const int *num_nodes,
                      	           const int *num_edges,
                                   void** list_node_feats, 
                                   void** list_edge_feats, 
                                   Dtype* node_input, 
                                   Dtype* edge_input);

extern "C" int PrepareIndices(const int num_graphs,
                              const int *num_nodes,
                              const int *num_edges,
                              void **list_of_edge_pairs, 
                              long long* edge_to_idx,
                              long long* edge_from_idx,
                              long long* g_idx);

#endif