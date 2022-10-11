This repository contains the necessary code and documents the results of studying several methods of explainable AI (xAI) on established toptagger models. We study the `TopoDNN` model, model based on `Multi-Body N Subjettiness (MBNS)` data, and the `Particle Flow Network (PFN)` model. The architecture and implementation of these models (and a few more that we will  investigate and add here in the near future) are reviewed and compared in the paper titled **The Machine Learning landscape of top taggers** by Kasieczka et al. (DOI: http://dx.doi.org/10.21468/SciPostPhys.7.1.014)

# Preparing the repository
To train/retrain models and run the notebooks provided with this repository, one needs to create the right environment and install certain dependencies. First `cd` to the project's top directory and do

`export PROJPATH=$PWD`

### Download Data
The dataset used by these studies has been made publicly available by Butter et al in **Deep-learned Top Tagging with a Lorentz Layer** (DOI: http://dx.doi.org/10.21468/SciPostPhys.5.3.028) and used for the review studies in the earlier mentioned review paper by Kasieczka et al. It is available at https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ6. The following steps need to be executed to download the data in store it in a location for processing.

```
cd $PROJPATH
mkdir -p datasets
cd datasets
wget https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ6/download
unzip download
mv v0/* .
rm download 
cd $PROJPATH
```

### Make necessary directories
Run the following commands to make the necessary directories
```
cd $PROJPATH
mkdir -p datasets/topoprocessed datasets/n-subjettiness_data 
mkdir -p evaluation/TopoDNN/figures evaluation/Multi-Body/figures evaluation/PFN/figures
mkdir -p models/TopoDNN/trained_models models/TopoDNN/trained_model_dicts
mkdir -p models/Multi-Body/trained_models models/Multi-Body/trained_model_dicts
mkdir -p models/PFN/trained_models models/PFN/trained_model_dicts
```

### Setup necessary environment! 
This step required anaconda installation. Please visit https://anaconda.org/ and follow the instructions there to install and setup `Anaconda`. Then run the following commands:
```
cd $PROJPATH
conda create --name toptagger_env --file requirements.txt
conda activate toptagger_env
pip install tensorflow
```
*Note:* The environment `toptagger_env` sets up necessary dependencies for using `PyTorch`. We found that asking conda to install `tensorflow` in the same environment can be troublesome and hence, a simpler solution of using `pip` to install it has been adapted.

### Data pre-processing
To preprocess the data for the `TopoDNN` and `MBNS` models, necessary scripts are given within the `datasets` directory. For the `TopoDNN` preprocessing, run the following commands

```
cd $PROJPATH/datasets
python topodnnpreprocessing.py <datasetname>
```
where `datasetname` can be either `train.py`, `val.py`, or `test.py`. The preprocessed data will be stored in the `topoprocessed` subdirectory. For the `MBNS` model, you will need to install the `FastJet` package. 

```
cd $PROJPATH
curl -O http://fastjet.fr/repo/fastjet-3.4.0.tar.gz
tar zxvf fastjet-3.4.0.tar.gz
cd fastjet-3.4.0/
./configure --prefix=$PROJPATH/fastjet-install --enable-pyext
make
make check
make install
cd $PROJPATH

PYPATH=`ls -d $PROJPATH/fastjet-install/lib/python3*`
export PYTHONPATH=$PYTHONPATH:$PYPATH/site-packages
```
and then simply do

```
cd $PROJPATH/datasets
python mb8spreprocessing.py
```
The preprocessed data will be stored in the `n-subjettiness_data` subdirectory.

# Training your own models
For each model architecture, we have trained a number of alternate variants and they are hosted in the `models/<model-type>/trained_models` directories where `model-type` can be `TopoDNN`, `Multi-Body`, or `PFN`. The necessary metadata for each model is given as `json` files in the `models/<model-type>/trained_model_dicts` directories. If you are interested in training your own models, please follow the instructions in the `README` file given within each `<model-type>` subdirectory within the `models` directory.

# Reproducing xAI results
The studies associated with explainability of each model are recorded in notebooks hosted in `evaluation/<model-type>` subdirectories. Each notebook is self-contained but **they rely on avaliability of the pretrained models and the datasets in the way they have been setup in the previous section**. The content of each notebook is explained in the `README` file provided in `evaluation/` directory.

# Reference
The studies in this repository are compiled and explained in this paper: [A Detailed Study of Interpretability of Deep Neural Network based Top Taggers](https://arxiv.org/abs/2210.04371)

To cite this work, please add-
```
A Khot, MS Neubauer, A Roy. A Detailed Study of Interpretability of Deep Neural Network based Top Taggers. arxiv preprint arXiv:2210.04371.
```
or use the following `bibtex` entry-
```
@article{khot2022detailed,
  title={A Detailed Study of Interpretability of Deep Neural Network based Top Taggers},
  author={Ayush Khot, Mark S. Neubauer, Avik Roy},
  journal={arXiv preprint arXiv:2210.04371},
  year={2022}
}
```

# Source Code
Existing resources from publicly available repositories have been adapted to implement the data preprocessing and training code. We are greatly thankful to the authors of the following works for making these repositories and resources publicly available. 

- Preprocessing and training for `TopoDNN` have been largely adopted from: https://github.com/jpearkes/topo_dnn/blob/master/topo_dnn.ipynb
- Network implementation for `PFN` has been obtained from: https://github.com/jet-universe/particle_transformer/blob/main/networks/example_PFN.py
- Implementation of Layerwise Relevance Propagation (LRP) is inspired from: https://git.tu-berlin.de/gmontavon/lrp-tutorial
- Part of preprocessing code for `MBNS` is obtained from: https://github.com/SebastianMacaluso/TreeNiN/blob/master/code/top_reference_dataset/ReadData.py

# Contact:
For comments, feedback, and suggestions: Avik Roy (avroy@illinois.edu) and Ayush Khot (akhot2@illinois.edu)
