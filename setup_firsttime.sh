## download data


## make necessary directories
mkdir -p datasets/topoprocessed datasets/multi_body_tau_data  datasets/n-subjettiness_data  datasets/PFN_latent_space_data  datasets/pnprocessed  
mkdir -p evaluation/TopoDNN/figures evaluation/TopoDNN/shap_info
mkdir -p evaluation/Multi-Body/figures
mkdir -p models/TopoDNN/trained_models models/TopoDNN/trained_model_dicts
mkdir -p models/Multi-Body/trained_models models/Multi-Body/trained_model_dicts

## get fastjet
curl -O http://fastjet.fr/repo/fastjet-3.4.0.tar.gz
tar zxvf fastjet-3.4.0.tar.gz
cd fastjet-3.4.0/
./configure --prefix=$PWD/../fastjet-install --enable-pyext
make
make check
make install
cd ..

export PYTHONPATH=$PYTHONPATH:$PWD/fastjet-install/lib/python3.8/site-packages