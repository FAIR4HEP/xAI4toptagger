To train a `TopoDNN` model instance- setup the required environment and run the following command-

```
python topodnntrain.py <options>
```

The available options are:

```
  -h, --help            show this help message and exit
  --outdir OUTDIR       Output directory for trained model
  --outdictdir OUTDICTDIR
                        Output directory for trained model metadata
  --nodes NODES         Comma-separated list of hidden layer nodes
  --epoch EPOCH         Epochs
  --label LABEL         a label for the model
  --batch-size BATCH_SIZE
                        batch_size
  --data-loc DATA_LOC   Directory for data
  --drop-pt0            Drop pt0 from training data
  --drop-pt             Drop all pt from training data
  --standardize-pt      Standard scalar stanadrization for all pt
  --nconst NCONST       Number of constituents to consider
```
