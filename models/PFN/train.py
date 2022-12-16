import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_pfn import PFNDataset, processed2tau
from pfn_model import ParticleFlowNetwork as Model
from augmented_pfn_model import AugParticleFlowNetwork as AugModel
import numpy as np
from sklearn.metrics import accuracy_score
from torchinfo import summary
import torch.nn as nn
import argparse, os, json

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--outdir", type=str, action="store", dest="outdir", default="./trained_models/", help="Output directory for trained model" )
    parser.add_argument("--outdictdir", type=str, action="store", dest="outdictdir", default="./trained_model_dicts/", help="Output directory for trained model metadata" )
    parser.add_argument("--Phi-nodes", type=str, action="store", dest="phi_nodes", default="100,100,256", help="Comma-separated list of hidden layer nodes for Phi")
    parser.add_argument("--F-nodes", type=str, action="store", dest="f_nodes", default="100,100,100", help="Comma-separated list of hidden layer nodes for F")
    parser.add_argument("--epochs", type=int, action="store", dest="epochs", default=50, help="Epochs")
    parser.add_argument("--label", type=str, action="store", dest="label", default="", help="a label for the model")
    parser.add_argument("--batch-size", type=int, action="store", dest="batch_size", default=250, help="batch_size")
    parser.add_argument("--data-loc", type=str, action="store", dest="data_loc", default="../../datasets/", help="Directory for data" )
    parser.add_argument("--preprocessed", action="store_true", dest="preprocessed", default=False, help="Use Preprocessing on Data")
    parser.add_argument("--augmented", action="store_true", dest="augmented", default=False, help="Use Preprocessing on Augmented Model")
    
    args = parser.parse_args()
    #seed_everything(42)
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    if not os.path.exists(args.outdictdir):
        os.mkdir(args.outdictdir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
    extra_name = args.label
    if extra_name != "" and not extra_name.startswith('_'):
        extra_name = '_' + extra_name
    if args.augmented and "aug" not in extra_name:
        extra_name += '_aug'
        
    model_dict = {}
    for arg in vars(args):
        model_dict[arg] = getattr(args, arg)
    f_model = open("{}/PFN{}.json".format(args.outdictdir, extra_name), "w")
    json.dump(model_dict, f_model, indent=3)
    f_model.close()


    #Parameters to change
    epochs = args.epochs
    preprocessed = args.preprocessed
    #used to differentiate different models

    #LR Scheduler
    use_lr_schedule = True
    milestones=[10, 20]
    gamma=0.1

    #optimizer parameters
    l_rate = 3e-4 
    opt_weight_decay = 0

    #Early stopping parameters
    early_stopping = True
    min_epoch_early_stopping = 20
    patience=10 #num_epochs if val acc doesn't change by 'tolerance' for 'patience' epochs.
    tolerance=1e-5


    #Loading training and validation datasets
    train_path = args.data_loc + "/train.h5"
    val_path = args.data_loc + "/val.h5"
    train_set = PFNDataset(train_path, preprocessed)
    val_set = PFNDataset(val_path, preprocessed)

    features = 3

    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                             num_workers=1, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, 
                            num_workers=1, pin_memory=True, persistent_workers=True)

    if args.augmented:
        all_data_train   = []
        all_masks_train  = []
        all_labels_train = []
        all_augdat_train = []
        all_taus_train   = []
        all_data_val   = []
        all_masks_val  = []
        all_labels_val = []
        all_augdat_val = []
        all_taus_val = []
        print("Storing all Training data!")
        for x,m,y,a in tqdm(trainloader):
            all_data_train.append(x)
            all_masks_train.append(m)
            all_labels_train.append(y)
            taus = processed2tau(x,a,preprocessed=preprocessed)
            all_taus_train.append(taus[:,:7])
        print("Storing all Validation data!")
        for x,m,y,a in tqdm(val_loader):
            all_data_val.append(x)
            all_masks_val.append(m)
            all_labels_val.append(y)
            taus = processed2tau(x,a,preprocessed=preprocessed)
            all_taus_val.append(taus[:,:7])
        

    if not args.augmented:
        model = Model(input_dims = features,
                      Phi_sizes = list(map(int, args.phi_nodes.split(','))),
                      F_sizes   = list(map(int, args.f_nodes.split(','))) ).to(device)
        summary(model, ((1, 200, features), (1, 1, 200)))
    else:
        print("Augmented PFN is being used!")
        model= AugModel(input_dims = features,
                        Phi_sizes = list(map(int, args.phi_nodes.split(','))),
                        F_sizes   = list(map(int, args.f_nodes.split(','))),
                        aug_size = 7).to(device)
        summary(model, ((1, 200, features), (1, 1, 200), (1,7)))

    # loss func and opt
    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(),  lr=l_rate, weight_decay=opt_weight_decay)

    if use_lr_schedule:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones, gamma=gamma)


    best_val_acc = 0
    no_change=0
    pre_val_acc = 0
    for epoch in range(epochs):
        print('Epoch ' + str(epoch))
        val_loss_total = 0
        train_loss_total = 0
        train_top1_total = 0
        val_top1_total = 0
        #train loop
        model.train()
        if args.augmented:
            trainloader = zip(all_data_train,all_masks_train,all_labels_train,all_taus_train)
            val_loader = zip(all_data_val,all_masks_val,all_labels_val,all_taus_val)
        for x,m,y,a in tqdm(trainloader):

            opt.zero_grad()
            x = x.to(device)
            m = m.to(device)
            y = y.to(device)
            a = a.to(device)

            if args.augmented:
                #taus = processed2tau(x,a,preprocessed).to(device)
                pred = model(x,m,a)
            else:
                pred = model(x,m)
            loss = crit(pred, y)

            train_loss_total += loss.item()
            #accuracy is determined by rounding. Any number <= 0.5 get's rounded down to 0
            #The rest get rounded up to 1
            with torch.no_grad():
                top1 = accuracy_score(pred[:,1].round().cpu(), y[:,1].cpu(), normalize=False)
                train_top1_total += top1.item()
            loss.backward()
            opt.step()

        #Early stopping after at least 20 epochs
        model.eval()
        with torch.no_grad():
            for x,m,y, a in tqdm(val_loader):
                x = x.to(device)
                m = m.to(device)
                y = y.to(device)
                a = a.to(device)

                if args.augmented:
                    #taus = processed2tau(x,a,precprocessed).to(device)
                    pred = model(x,m,a)
                else:
                    pred = model(x,m)
                loss = crit(pred, y)

                val_loss_total += loss.item()
                #accuracy is determined by rounding. Any number <= 0.5 get's rounded down to 0
                #The rest get rounded up to 1
                with torch.no_grad():
                    top1 = accuracy_score(pred[:,1].round().cpu(), y[:,1].cpu(), normalize=False)
                    val_top1_total += top1.item()
        train_loss_total /= len(train_set)
        val_loss_total /= len(val_set)
        val_top1_total /= len(val_set)
        train_top1_total /= len(train_set)

        print('Best Validation Accuracy: ' + str(best_val_acc))
        print('Current Validation Accuracy: ' + str(val_top1_total))
        print('Current Validation Loss: ' + str(val_loss_total))

        if early_stopping:
            if abs(pre_val_acc - val_top1_total) < tolerance and epoch >= min_epoch_early_stopping:
                no_change+=1
                print('Validation Accuracy has not changed much, will stop in ' + str(patience-no_change) + 
                      ' epochs if this continues')
                if no_change==patience:
                    print('Stopping training')
                    break
            if abs(pre_val_acc - val_top1_total) >= tolerance and epoch >= min_epoch_early_stopping:
                no_change = 0

        if val_top1_total > best_val_acc:
            no_change=0
            print('Saving best model based on accuracy')
            torch.save(model.state_dict(), args.outdir + '/PFN_best'+extra_name)
            best_val_acc = val_top1_total

        #pre_val_loss = val_loss_total
        pre_val_acc = val_top1_total

        if use_lr_schedule:
            scheduler.step()


    print('Saving last model')
    torch.save(model.state_dict(), args.outdir + '/PFN_last'+extra_name)

