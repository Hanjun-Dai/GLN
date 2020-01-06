#ifndef cfg_H
#define cfg_H

#include <iostream>
#include <cstring>
#include <fstream>
#include <set>
#include <map>

typedef float Dtype;

struct cfg
{
    static int num_atom_types;
    static int nodefeat_dim;
    static int edgefeat_dim;

    static void LoadParams(const int argc, const char** argv)
    {
        for (int i = 1; i < argc; i += 2)
        {
            if (strcmp(argv[i], "-num_atom_types") == 0)
                num_atom_types = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-nodefeat_dim") == 0)
                nodefeat_dim = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-edgefeat_dim") == 0)
                edgefeat_dim = atoi(argv[i + 1]);                
        }
        std::cerr << "====== begin of gnn_clib configuration ======" << std::endl;
        std::cerr << "| num_atom_types = " << num_atom_types << std::endl;
        std::cerr << "| nodefeat_dim = " << nodefeat_dim << std::endl;
        std::cerr << "| edgefeat_dim = " << edgefeat_dim << std::endl;
        std::cerr << "======   end of gnn_clib configuration ======" << std::endl;
    }
};

#endif