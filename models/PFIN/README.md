To train a `PFIN` model instance- setup the required environment and run the following command-

```
python train.py <options>
```

The available options are:

```
  -h, --help            show this help message and exit
  --outdir		Output directory for trained model
  --outdictdir		Output directory for trained model metadata
  --Np			Number of constituents to consider
  --Phi-nodes		Comma-separated list of hidden layer nodes for Phi
  --NPhiI		Number of nodes in hidden layers for PhiI and PhiI2 
  --F-nodes		Comma-separated list of hidden layer nodes for F
  --epochs		Number of epochs
  --x-mode		Mode of Interaction: ['sum', 'cat']
  --label		a label for the model
  --batch-size		Batch size
  --data-loc		Directory for data
  --preprocessed        Use Preprocessing on Data
  --augmented		Use Augmented PFIN model (use the tau_x_1 variables to augment the latent space)
  --preload             Preload weights and biases from a pre-trained PFN/PFIN Model
  --preload-file        Location of the model to the preload
```