The different notebooks provided under different subdirectories contain the explainability studies performed for each of the different models. The content of each notebooks is summarized below.

## TopoDNN

- **TopoDNN_FeatureHists.ipynb**: Obtain some of the feature histograms
- **TopoDNN_evaluation.ipynb**: Contains basic evaluation of the baseline and variant models by calculating their ROC-AUC score as well as accuracies
- **TopoDNN_Shap.ipynb**: Performs feature ranking using SHAP (SHapely Additive exPlanations)
- **TopoDNN_dAUC.ipynb**: Performs feature ranking using $\Delta$AUC score and MAD (Mean Absolute Differential) Relevance scores
- **TopoDNN_LRP.ipynb**: Performs Layerwise Relevanace Propagation (LRP) for the TopoDNN model. Also contains code to obtain the feature correlation matrices for different jet categories
- **TopoDNN_NAP.ipynb**: Performs calculation of Relative Neural Activation (RNA) score along with Neural Activation Pattern (NAP) diagrams

## Multi-Body

- **MultiBody-FeatureHists.ipynb**: Obtain some of the feature histograms
- **MultiBody-Evaluation.ipynb**: Contains basic evaluation of the baseline and variant models by calculating their ROC-AUC score as well as accuracies
- **MultiBody-Shap.ipynb**: Performs feature ranking using SHAP 
- **MultiBody-dAUC.ipynb**: Performs feature ranking using $\Delta$AUC score and MAD  Relevance scores
- **MultiBody-LRP.ipynb**: Performs LRP for the TopoDNN model. Also contains code to obtain the feature correlation matrices for different jet categories
- **MultiBody-NAP.ipynb**: Performs calculation of RNA scores along with NAP diagrams


## PFN

- **PFN-Evaluation.ipynb**: Contains basic evaluation of the baseline and variant models by calculating their ROC-AUC score as well as accuracies
- **PFN-Explorer.ipynb**: Performs feature ranking using $\Delta$AUC score and MAD Relevance scores for input and latent space features. Obtains latent space feature histograms and correlation matrics
- **MultiBody-NAP.ipynb**: Performs calculation of RNA scores along with NAP diagrams
- **MultiBody-PCA.ipynb**: Performs Principal Component Analysis (PCA) on the latent space
