#include "mol_utils.h"
#include "config.h"

#include <assert.h>

void MolFeat::ParseAtomFeat(Dtype* arr, int feat)
{
    // atom_idx_map
    int t = feat & ((1 << 8) - 1);
    arr[t] = 1.0;
    feat >>= 8;		
    int base_idx = cfg::num_atom_types + 1;

    // getDegree
    int mask = (1 << 4) - 1;
    t = feat & mask;
    arr[base_idx + t] = 1.0;
    feat >>= 4;
    base_idx += 8;

    // getTotalNumHs
    t = feat & mask;
    arr[base_idx + t] = 1.0;
    feat >>= 4;
    base_idx += 6;

    // getImplicitValence
    t = feat & mask;
    arr[base_idx + t] = 1.0;
    feat >>= 4;
    base_idx += 7;

    // getIsAromatic
    if (feat & mask)
        arr[base_idx] = 1.0;
    assert(base_idx + 1 == cfg::nodefeat_dim);
}


void MolFeat::ParseEdgeFeat(Dtype* arr, int feat)
{
    int mask = (1 << 8) - 1;
    // getBondType
    arr[feat & mask] = 1.0;
    feat >>= 8;		
    // getIsConjugated
    if (feat & mask)
        arr[4] = 1.0;
    feat >>= 8;		
    // is ring
    int t = feat & mask;
    if (t == 2)
        arr[6] = 1.0;
    else if (feat & mask)
        arr[5] = 1.0;
}
