import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from multidataset import MultiDataset
from multimodel import Net as Model
import numpy as np
from sklearn.metrics import accuracy_score
from torchinfo import summary
import argparse
import json, os, sys

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
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--outdir", type=str, action="store", dest="outdir", default="./trained_models/", help="Output directory for trained model" )
    parser.add_argument("--outdictdir", type=str, action="store", dest="outdictdir", default="./trained_model_dicts/", help="Output directory for trained model metadata" )
    parser.add_argument("--nodes", type=str, action="store", dest="nodes", default="200,200,50,50", help="Comma-separated list of hidden layer nodes")
    parser.add_argument("--epochs", type=int, action="store", dest="epochs", default=200, help="Epochs")
    parser.add_argument("--label", type=str, action="store", dest="label", default="", help="a label for the model")
    parser.add_argument("--batch-size", type=int, action="store", dest="batch_size", default=256, help="batch_size")
    parser.add_argument("--data-loc", type=str, action="store", dest="data_loc", default="../../datasets/n-subjettiness_data/", help="Directory for data" )
    parser.add_argument("--use-pt", action="store_true", dest="use_pt", default=False, help="Use Jet pT")
    parser.add_argument("--use-mass", action="store_true", dest="use_mass", default=False, help="Use Jet mass")
    parser.add_argument("--tau-x-1", action="store_true", dest="tau_x_1", default=False, help="Use tau_x_1 variables alone")
    parser.add_argument("-N", type=int, action="store", dest="N", default=8, help="Order of subjettiness variables")
    
    args = parser.parse_args()
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    if not os.path.exists(args.outdictdir):
        os.mkdir(args.outdictdir)
    
    #I train on seed 42
    seed_everything(42)

    use_jet_pt = args.use_pt
    use_jet_mass = args.use_mass
    tau_x_1 = args.tau_x_1
    N = args.N
    epochs = args.epochs
    batch_size = args.batch_size
    hidden = list(map(int, args.nodes.split(',')))

    #used to differentiate different models
    extra_name = args.label
    if extra_name != '' and not extra_name.startswith('_'):
        extra_name = '_' + extra_name
    if tau_x_1 and 'tau_x_1' not in extra_name:
        extra_name += '_tau_x_1'
    
    model_dict = {}
    for arg in vars(args):
        model_dict[arg] = getattr(args, arg)
    f_model = open("{}/MultiBody{}-Subjettiness_mass".format(args.outdictdir, N) + str(use_jet_mass) + '_pt' + str(use_jet_pt) + extra_name + ".json", "w")
    json.dump(model_dict, f_model, indent=3)
    f_model.close()


    
    #LR Scheduler
    use_lr_schedule = True
    milestones=[75,150,500]
    gamma=0.1

    #optimizer parameters
    l_rate = 3e-4
    opt_weight_decay = 0

    #Early stopping parameters
    early_stopping = True
    min_epoch_early_stopping = 100
    patience=10 #num_epochs if val acc doesn't change by 'tolerance' for 'patience' epochs.
    tolerance=1e-5

    #Training and validation paths
    train_path = args.data_loc + '/train_all.npy'
    val_path = args.data_loc + '/val_all.npy'

    #Loading training and validation datasets
    train_set = MultiDataset(train_path, N, use_jet_pt, use_jet_mass, tau_x_1)
    val_set = MultiDataset(val_path, N, use_jet_pt, use_jet_mass, tau_x_1)

    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    # model
    model = Model(N, use_jet_pt, use_jet_mass, tau_x_1, hidden).cuda()
    #num of features
    features = 3*(N-1)-1
    if tau_x_1:
        features = N-1
    if use_jet_mass:
        features+=1
    if use_jet_pt:
        features+=1
    summary(model, (1, features))
    model = model.double()

    # loss func and opt
    crit = torch.nn.BCELoss()
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
        for x,y in tqdm(trainloader):

            opt.zero_grad()
            x = x.cuda()
            y = y.cuda()

            pred = model(x)
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
            for x,y in tqdm(val_loader):
                x = x.cuda()
                y = y.cuda()
                pred = model(x)
                loss = crit(pred, y)

                val_loss_total += loss.item()
                #accuracy is determined by rounding. Any number <= 0.5 get's rounded down to 0
                #The rest get rounded up to 1
                top1 = accuracy_score(pred[:,1].round().cpu(), y[:,1].cpu(), normalize=False)
                val_top1_total += top1.item()

        val_loss_total /= len(val_loader)
        train_loss_total /= len(trainloader)
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
            torch.save(model.state_dict(), args.outdir + '/MultiBody' + str(N) + '-Subjettiness_mass' +str(use_jet_mass)+'_pt'+str(use_jet_pt)+'_best'+extra_name)
            best_val_acc = val_top1_total

        #pre_val_loss = val_loss_total
        pre_val_acc = val_top1_total

        if use_lr_schedule:
            scheduler.step()


    print('Saving last model')
    torch.save(model.state_dict(), args.outdir + '/MultiBody' + str(N) + '-Subjettiness_mass' +str(use_jet_mass)+'_pt'+str(use_jet_pt)+'_last'+extra_name)

