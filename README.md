
## download data
mkdir -p datasets
cd datasets
wget https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ6/download
unzip download
mv v0/* .
rm download 
cd -


## setup necessary environment! This step required anaconda installation
conda create --name toptagger_env --file requirements.txt
conda activate toptagger_env
pip install tensorflow

## get fastjet
curl -O http://fastjet.fr/repo/fastjet-3.4.0.tar.gz
tar zxvf fastjet-3.4.0.tar.gz
cd fastjet-3.4.0/
./configure --prefix=$PWD/../fastjet-install --enable-pyext
make
make check
make install
cd ..

PYPATH=`ls -d $PWD/fastjet-install/lib/python3*`
export PYTHONPATH=$PYTHONPATH:$PYPATH/site-packages


## make necessary directories
mkdir -p datasets/topoprocessed datasets/n-subjettiness_data 
mkdir -p evaluation/TopoDNN/figures evaluation/Multi-Body/figures evaluation/PFN/figures
mkdir -p models/TopoDNN/trained_models models/TopoDNN/trained_model_dicts
mkdir -p models/Multi-Body/trained_models models/Multi-Body/trained_model_dicts
mkdir -p models/PFN/trained_models models/PFN/trained_model_dicts