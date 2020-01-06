# GLN
Implementation of Retrosynthesis Prediction with Conditional Graph Logic Network

https://papers.nips.cc/paper/9090-retrosynthesis-prediction-with-conditional-graph-logic-network

# Install

This package requires the **rdkit** and **pytorch**.

For rdkit, we tested with [2019_03_3](https://github.com/rdkit/rdkit/releases/tag/Release_2019_03_3). Please build the package from source and setup the `PYTHONPATH` and `LD_LIBRARY_PATH` properly, as this package requires the dynamic lib built from rdkit.

For pytorch, we tested from 1.2.0 to 1.3.1. Please don't use versions older than that, as this package contains c++ customized ops which relies on specific toolchains from pytorch.

After the above preparation, simply navigate to the project root folder and install:

    cd GLN
    pip install -e .

Note that by default the cuda ops will not be enabled on Mac OSX.

# Reference

If you find our paper/code is useful, please consider citing our paper:

    @inproceedings{dai2019retrosynthesis,
      title={Retrosynthesis Prediction with Conditional Graph Logic Network},
      author={Dai, Hanjun and Li, Chengtao and Coley, Connor and Dai, Bo and Song, Le},
      booktitle={Advances in Neural Information Processing Systems},
      pages={8870--8880},
      year={2019}
    }
    
The orignal version of rdchiral comes from https://github.com/connorcoley/rdchiral, and this repo has made a wrapper over that. Please also cite the corresponding paper if possible:

    @article{coley2019rdchiral,
      title={RDChiral: An RDKit Wrapper for Handling Stereochemistry in Retrosynthetic Template Extraction and Application},
      author={Coley, Connor W and Green, William H and Jensen, Klavs F},
      journal={Journal of chemical information and modeling},
      publisher={ACS Publications}
    }
