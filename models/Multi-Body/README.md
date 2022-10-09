To train a `MBNS` model instance- setup the required environment and run the following command-

```
python train.py <options>
```

The available options are:
```
  -h, --help              show this help message and exit
  --outdir OUTDIR         Output directory for trained model
  --outdictdir OUTDICTDIR Output directory for trained model metadata
  --nodes NODES           Comma-separated list of hidden layer nodes
  --epochs EPOCHS         Epochs
  --label LABEL           a label for the model
  --batch-size BATCH_SIZE batch_size
  --data-loc DATA_LOC     Directory for data
  --use-pt                Use Jet pT
  --use-mass              Use Jet mass
  --tau-x-1               Use tau_x_1 variables alone
  -N N                    Order of subjettiness variables
```
