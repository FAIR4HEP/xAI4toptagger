To train a `PFN` model instance- setup the required environment and run the following command-

```
python train.py <options>
```

The available options are:

```
  -h, --help            show this help message and exit
  --outdir OUTDIR       Output directory for trained model
  --outdictdir OUTDICTDIR
                        Output directory for trained model metadata
  --Phi-nodes PHI_NODES
                        Comma-separated list of hidden layer nodes for Phi
  --F-nodes F_NODES     Comma-separated list of hidden layer nodes for F
  --epochs EPOCHS       Epochs
  --label LABEL         a label for the model
  --batch-size BATCH_SIZE
                        batch_size
  --data-loc DATA_LOC   Directory for data
  --preprocessed        Use Preprocessing on Data

```
