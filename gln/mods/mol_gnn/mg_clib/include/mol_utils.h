#ifndef MOL_UTILS_H
#define MOL_UTILS_H

#include <vector>
#include <map>

#include "config.h"

struct MolFeat
{
	static void InitIdxMap();
	
	static void ParseAtomFeat(Dtype* arr, int feat);
	
	static void ParseEdgeFeat(Dtype* arr, int feat);
	
	static std::map<unsigned, unsigned> atom_idx_map;
};


#endif