To train a `PFN` model instance- setup the required environment and run the following command-

```
python train.py <options>
```

The available options are:

```
  -h, --help            show this help message and exit
  --outdir		Output directory for trained model
  --outdictdir		Output directory for trained model metadata
  --Phi-nodes		Comma-separated list of hidden layer nodes for Phi
  --F-nodes		Comma-separated list of hidden layer nodes for F
  --epochs		Number of epochs
  --label		a label for the model
  --batch-size		Batch size
  --data-loc		Directory for data
  --preprocessed        Use Preprocessing on Data

```