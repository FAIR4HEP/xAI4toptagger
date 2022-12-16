import os
import pandas as pd
import json
import keras 
import argparse
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import Adam



if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--outdir", type=str, action="store", dest="outdir", default="./trained_models/", help="Output directory for trained model" )
    parser.add_argument("--outdictdir", type=str, action="store", dest="outdictdir", default="./trained_model_dicts/", help="Output directory for trained model metadata" )
    parser.add_argument("--nodes", type=str, action="store", dest="nodes", default="300,102,12,6", help="Comma-separated list of hidden layer nodes")
    parser.add_argument("--epoch", type=int, action="store", dest="epoch", default=40, help="Epochs")
    parser.add_argument("--label", type=str, action="store", dest="label", default="", help="a label for the model")
    parser.add_argument("--batch-size", type=int, action="store", dest="batch_size", default=96, help="batch_size")
    parser.add_argument("--data-loc", type=str, action="store", dest="data_loc", default="../../datasets/topoprocessed/", help="Directory for data" )
    parser.add_argument("--drop-pt0", action="store_true", dest="drop_pt0", default=False, help="Drop pt0 from training data")
    parser.add_argument("--drop-pt", action="store_true", dest="drop_pt", default=False, help="Drop all pt from training data")
    parser.add_argument("--standardize-pt", action="store_true", dest="standardize_pt", default=False, help="Standard scalar stanadrization for all pt")
    parser.add_argument("--nconst", type=int, action="store", dest="nconst", default=30, help="Number of constituents to consider")

    args = parser.parse_args()
    mode = args.label
    if mode != "":
        mode = mode if mode.startswith("_") else ("_" + mode)

    # Load inputs
    df_train = pd.read_pickle(args.data_loc + "/train.pkl")
    df_val   = pd.read_pickle(args.data_loc + "val.pkl")
    x_train  = df_train.loc[:, df_train.columns != 'is_signal_new']
    y_train  = df_train["is_signal_new"]
    x_val    = df_val.loc[:, df_train.columns != 'is_signal_new']
    y_val    = df_val["is_signal_new"]
    x_train = x_train.iloc[:,:3*args.nconst]
    x_val = x_val.iloc[:,:3*args.nconst]
    del df_train
    del df_val


    if args.drop_pt0:
        #Get rid of pt_0 column
        x_train = x_train.loc[:, x_train.columns != 'pt_0']
        x_val = x_val.loc[:, x_val.columns != 'pt_0']
        if '_pt0' not in mode.strip('_'):
            mode = mode.strip('_') + '_pt0'
    if args.drop_pt:
        #Get rid of all pt
        pt_cols = [col for col in x_train.columns if 'pt' in col]
        x_train = x_train.drop(pt_cols, axis=1)
        x_val = x_val.drop(pt_cols, axis=1)
        if '_pt' not in mode.strip('_'):
            mode = mode.strip('_') + '_pt'
    if args.standardize_pt:
        pt_cols = [col for col in x_train.columns if 'pt' in col]
        x_train[pt_cols] = (x_train[pt_cols] - x_train[pt_cols].mean())/x_train[pt_cols].std()
        x_val[pt_cols] = (x_val[pt_cols] - x_val[pt_cols].mean())/x_val[pt_cols].std()
        if '_standardize_pt' not in mode.strip('_'):
            mode = mode.strip('_') + '_standardize_pt'
    
    args.label = mode
    model_dict = {}
    for arg in vars(args):
        model_dict[arg] = getattr(args, arg)
    f_model = open("{}/topodnnmodel{}.json".format(args.outdictdir, mode), "w")
    json.dump(model_dict, f_model, indent=3)
    f_model.close()

    nodes = list(map(int, args.nodes.split(',')))
    model = Sequential()
    model.add(Dense(nodes[0], input_dim=x_train.shape[1]))
    model.add(Activation('relu'))
    for node in nodes[1:-1]:
        model.add(Dense(node))
        model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
                    x_train,
                    y_train,
                    batch_size=args.batch_size,
                    callbacks=[
                        EarlyStopping(
                            verbose=True,
                            patience=5,
                            monitor='val_loss'),
                        ModelCheckpoint(
                            'trained_models/topodnnmodel'+mode,
                            monitor='val_loss',
                            verbose=True,
                            save_best_only=True)],
                            epochs=40,
                            validation_data=(
                        x_val,
                        y_val))

