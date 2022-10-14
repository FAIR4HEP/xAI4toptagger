import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
                     
def get_pt_eta_phi_v(px, py, pz):
    '''Provides pt, eta, and phi given px, py, pz'''
    # Init variables
    pt = np.zeros(len(px))
    pt = np.sqrt(np.power(px,2) + np.power(py,2))
    phi = np.zeros(len(px))
    eta = np.zeros(len(px))
    theta = np.zeros(len(px))
    x = np.where((px!=0) | (py!=0) | (pz!=0)) # locate where px,py,pz are all 0 
    theta[x] = np.arctan2(pt[x],pz[x]) 
    cos_theta = np.cos(theta)
    y = np.where(np.power(cos_theta,2) < 1)
    eta[y] = -0.5*np.log((1 - cos_theta[y]) / (1 + cos_theta[y]))
    z = np.where((px !=0)|(py != 0))
    phi[z] = np.arctan2(py[z],px[z])
    return pt, eta, phi                     

class PFNDataset(Dataset):
    def __init__(self, file_path, preprocessed=True):
        
        # Read in hdf5 files
        store = pd.HDFStore(file_path)
        df = store.select("table")                     
        n_constits = 200 # use only 200 highest pt jet constituents 
        df_pt_eta_phi = pd.DataFrame()
        
        e_cols = ["E_{}".format(i) for i in range(200)]
        px_cols = ["PX_{}".format(i) for i in range(200)]
        py_cols = ["PY_{}".format(i) for i in range(200)]
        pz_cols = ["PZ_{}".format(i) for i in range(200)]
        jet_e = np.array(df[e_cols].sum(axis=1)).reshape(-1,1)
        jet_px = np.array(df[px_cols].sum(axis=1)).reshape(-1,1)
        jet_py = np.array(df[py_cols].sum(axis=1)).reshape(-1,1)
        jet_pz = np.array(df[pz_cols].sum(axis=1)).reshape(-1,1)
        jet_pt, jet_eta, jet_phi = get_pt_eta_phi_v(jet_px.flatten(), jet_py.flatten(), jet_pz.flatten())
        jet_pt = jet_pt.reshape(-1,1)
        jet_eta = jet_eta.reshape(-1,1)
        jet_phi = jet_phi.reshape(-1,1)
        jet_m = (np.abs(jet_e**2 - jet_px**2 - jet_py**2 -jet_pz**2))**0.5
        jet_nconst = np.array((df[e_cols] != 0).sum(axis=1)).reshape(-1,1)
        for j in range(n_constits):
            i = str(j)
            #print("Processing constituent #"+str(i))
            px = np.array(df["PX_"+i][0:])
            py = np.array(df["PY_"+i][0:])
            pz = np.array(df["PZ_"+i][0:])
            pt,eta,phi = get_pt_eta_phi_v(px,py,pz)
            df_pt_eta_phi_mini = pd.DataFrame(np.stack([pt,eta,phi]).T,columns = ["pt_"+i,"eta_"+i,"phi_"+i])
            df_pt_eta_phi = pd.concat([df_pt_eta_phi,df_pt_eta_phi_mini], axis=1, sort=False)
        df_pt_eta_phi = df_pt_eta_phi.astype('float32')   
        eta_cols = [col for col in df_pt_eta_phi.columns if 'eta' in col]
        phi_cols = [col for col in df_pt_eta_phi.columns if 'phi' in col]
        pt_cols = [col for col in df_pt_eta_phi.columns if 'pt' in col] 
        px_cols = [col for col in df.columns if 'PX' in col] 
        py_cols = [col for col in df.columns if 'PY' in col]
        pz_cols = [col for col in df.columns if 'PZ' in col]
        df_jet_pet_eta_phi = pd.DataFrame()
        df_jet_pet_eta_phi['jet_px'] = df[px_cols].sum(axis=1)
        df_jet_pet_eta_phi['jet_py'] = df[py_cols].sum(axis=1)
        df_jet_pet_eta_phi['jet_pz'] = df[pz_cols].sum(axis=1)
        #labels, prob_isQCD, prob_isSignal
        self.labels = np.expand_dims(df["is_signal_new"].to_numpy(), axis=0)
        self.labels = np.append(1-self.labels, self.labels, 0)
        
        del df
        if preprocessed:
            jet_px = np.array(df_jet_pet_eta_phi['jet_px'])
            jet_py = np.array(df_jet_pet_eta_phi['jet_py'])
            jet_pz = np.array(df_jet_pet_eta_phi['jet_pz'])
            del df_jet_pet_eta_phi
            jet_pt, jet_eta, jet_phi = get_pt_eta_phi_v(jet_px,jet_py,jet_pz)
            #Preprocessing
            df_pt_eta_phi[pt_cols]= df_pt_eta_phi[pt_cols].div(df_pt_eta_phi[pt_cols].sum(axis=1), axis=0)
            df_pt_eta_phi[eta_cols] = df_pt_eta_phi[eta_cols].subtract(pd.Series(jet_eta),axis=0)
            df_pt_eta_phi[phi_cols] = df_pt_eta_phi[phi_cols].subtract(pd.Series(jet_phi),axis=0)
            jet_pt, jet_eta, jet_phi = jet_pt.reshape(-1,1), jet_eta.reshape(-1,1), jet_phi.reshape(-1,1)
        self.aug_data = np.concatenate((jet_e, jet_m, jet_pt, jet_eta, jet_phi, jet_nconst), axis = 1)
        self.columns = df_pt_eta_phi.columns
        self.data = df_pt_eta_phi.to_numpy()
        self.mask = np.where(df_pt_eta_phi[pt_cols].to_numpy() != 0, 1, 0)
        self.labels = torch.from_numpy(self.labels).float()
        self.mask = torch.from_numpy(self.mask).float()
        self.data = torch.from_numpy(self.data).float()
        self.labels = self.labels.permute(1, 0)
        self.mask = self.mask.reshape(-1, 1, 200)
        self.data = torch.reshape(self.data, (-1, 200, 3))
        #gets rid of phi/eta subtraction from zero-vectors
        m=self.mask.reshape(-1,200,1)
        self.data = torch.cat((m,m,m), dim=2)*self.data
    def __len__(self):
        return len(self.mask)

    def __getitem__(self, idx):
        label = self.labels[idx]
        item = self.data[idx]
        mask = self.mask[idx]
        aug_data = self.aug_data[idx]
        return item, mask, label, aug_data
    
if __name__ == "__main__":
    preprocessed=True
    mydataset = PFNDataset("../../datasets/val.h5", preprocessed)
    print(len(mydataset))
    trainloader = DataLoader(mydataset, batch_size=500, shuffle=False, num_workers=40, pin_memory=True, persistent_workers=True)
    for i,m,l in trainloader:
        #print(i)
        #print(m)
        #print(l)
        print(i.shape)
        print(m.shape)
        print(l.shape)
        break
        
        #python dataset_pfn.py
