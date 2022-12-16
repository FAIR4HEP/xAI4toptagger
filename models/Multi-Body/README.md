To train a `MBNS` model instance- setup the required environment and run the following command-

```
python train.py <options>
```

The available options are:
```
  -h, --help              show the help message and exit
  --outdir		  Output directory for trained model
  --outdictdir		  Output directory for trained model metadata
  --nodes		  Comma-separated list of hidden layer nodes
  --epochs		  Number of epochs
  --label		  A label for the model
  --batch-size		  Batch size
  --data-loc		  Directory for data
  --use-pt                Use Jet pT
  --use-mass              Use Jet mass
  --tau-x-1               Use tau_x_1 variables alone
  -N 			  Order of subjettiness variables
```
