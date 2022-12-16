To train a `TopoDNN` model instance- setup the required environment and run the following command-

```
python topodnntrain.py <options>
```

The available options are:

```
  -h, --help            show this help message and exit
  --outdir		Output directory for trained model
  --outdictdir		Output directory for trained model metadata
  --nodes		Comma-separated list of hidden layer nodes
  --epoch		Epochs
  --label		a label for the model
  --batch-size		batch_size
  --data-loc		Directory for data
  --drop-pt0		Drop pt0 from training data
  --drop-pt		Drop all pt from training data
  --standardize-pt	Standard scalar stanadrization for all pt
  --nconst		Number of constituents to consider
```
