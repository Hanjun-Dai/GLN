# GLN
Implementation of Retrosynthesis Prediction with Conditional Graph Logic Network

https://papers.nips.cc/paper/9090-retrosynthesis-prediction-with-conditional-graph-logic-network

# Setup

- ## Install package

This package requires the **rdkit** and **pytorch**.

For rdkit, we tested with [2019_03_3](https://github.com/rdkit/rdkit/releases/tag/Release_2019_03_3). Please build the package from source and setup the `PYTHONPATH` and `LD_LIBRARY_PATH` properly, as this package requires the dynamic lib built from rdkit.

For pytorch, we tested from 1.2.0 to 1.3.1. Please don't use versions older than that, as this package contains c++ customized ops which relies on specific toolchains from pytorch.

After the above preparation, simply navigate to the project root folder and install:

    cd GLN
    pip install -e .

Note that by default the cuda ops will not be enabled on Mac OSX.

- ## Dropbox

We provide the raw datasets, the cooked (data after preprocessing) datasets, and also the trained model dumps in a dropbox folder. 

`https://www.dropbox.com/sh/6ideflxcakrak10/AADTbFBC0F8ax55-z-EDgrIza?dl=0`

The cooked dataset is pretty large. You can also simply download the raw datasets only, and use the script provided in this repo for preprocessing. You don't have to have a dropbox to download, and the result folder doesn't have to be in your dropbox. The only thing needed is to create a symbolic link named `dropbox` and put it in the right place.

Finally the folder structure will look like this: 

```
GLN
|___gln  # source code
|   |___common # common implementations
|   |___...
|
|___setup.py 
|
|___dropbox  # data, trained model dumps and cooked data, this can be a symbolic link
    |___schneider50k # raw data
    |___|__raw_train.csv
    |   |__...
    |
    |___cooked_schneider50k # cooked data
    |___schneider50k.ckpt # trained model dump
...
```

# Preprocessing

If you download the cooked data in the previous step, you can simply skip this step.

Our paper mainly focused on schneider50k dataset. The raw data and data split is the same as https://github.com/connorcoley/retrosim/blob/master/retrosim/data/get_data.py

Note that for both the reaction type unknown and type given experiments, we go throught the same steps as below. The only difference is the dataset name: **schneider50k** (type unknown) v.s. **typed_schneider50k** (type given)

First go to the data processing folder:
```
cd gln/data_process
```
Then run the script in the following order:

- **Get canonical smiles**

Please specify the dataset name accordingly
```
./step0.0_run_get_cano_smiles.sh
```

- **Extract raw templates**

Please specify the dataset name, as well as #cpu threads (the more the better). 
```
./step0.1_run_raw_template_extract.sh
```

1. **Filter template**

One can filter out uncommon templates using this script. We didn't filter out any template in our paper, though a careful selection of templates may further improve the performance.

Please specify the dataset name, template name (arbitrary one is fine), and possibly the minimum number of occurance one template needs to have in order to be included (default=1)

```
./step1_filter_template.sh
```



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
