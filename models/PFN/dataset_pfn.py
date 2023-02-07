import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import sys,os
sys.path.append("../../datasets/")
sys.path.append(os.path.abspath('../../fastjet-install/lib/python3.9/site-packages'))
from PreProcessTools import *
try:
    from fastjet import *
except:
    print("FastJet not found. Make sure FastJet has been installed and the PYTHONPATH variable includes the path to FastJet package library")
    sys.exit(1)


def processed2tau(x,a,N=8,preprocessed=True):
    # x = preprocessed data of shape (-1, 200, 3) (pt, eta, phi)
    # a = augmented data: (jet_e, jet_m, jet_pt, jet_eta, jet_phi, jet_ptsum, jet_nconst)
    
    x2 = torch.clone(x).detach().float().to(x.device)
    a2 = torch.clone(a).detach().float().to(a.device)
    jet_pt = a2[:,2].reshape(-1,1)
    jet_ptsum = a2[:,5].reshape(-1,1)
    jet_nconst = a2[:,6].reshape(-1,1)
    jet_eta = a2[:,3].reshape(-1,1)
    jet_phi = a2[:,4].reshape(-1,1)

    if preprocessed:
        x2[:,:,0] = x2[:,:,0]*jet_ptsum
        x2[:,:,1] += jet_eta
        x2[:,:,2] += jet_phi

    # now x2 should be real (pt, eta, phi) of each particle
    px = (x2[:,:,0]*torch.cos(x2[:,:,2])).reshape(-1,200,1)
    py = (x2[:,:,0]*torch.sin(x2[:,:,2])).reshape(-1,200,1)
    pz = (x2[:,:,0]*torch.sinh(x2[:,:,1])).reshape(-1,200,1)
    e = (px**2 + py**2 +pz**2)**0.5

    # get the 4 vector format
    part_ps = torch.cat((e,px,py,pz), 2)
    taus = torch.zeros(len(part_ps), N).float().to(part_ps.device)

    # Now calculate taus
    jet_def = JetDefinition(kt_algorithm, 0.4)
    badjets = []
    all_clusters = []
    all_jet_particle_etaphis = []
    all_jet_particle_pts = []

    for ii in range(len(part_ps)):
        if jet_nconst[ii,0] >= N:
            dummy_jet = torch.clone(part_ps[ii])
            ndummy = int(jet_nconst[ii,0])
            dummy_jet_particle_pts = ((dummy_jet[:,1]**2 + dummy_jet[:,2]**2)**0.5).reshape(-1)
            dummy_jet_particle_etaphis = torch.cat((x2[ii,:,2].reshape(-1,1), x2[ii,:,1].reshape(-1,1)), 1)
            dummy_jet_pt = jet_pt[ii,0]
            break
    
    for idx, jet in enumerate(part_ps):
        nconst = int(jet_nconst[idx,0])
        if nconst < N:
            badjets.append(idx)
            badjet = True
        else:
            badjet = False

        if not badjet:
            this_particles = [PseudoJet(px.item(), py.item(), pz.item(), e.item()) for (e, px, py, pz) in jet[:nconst,:]]
            all_clusters.append(ClusterSequence(this_particles, jet_def))
            this_jet_particle_pts = ((jet[:,1]**2 + jet[:,2]**2)**0.5).reshape(-1)
            this_jet_particle_etaphis = torch.cat((x2[idx,:,2].reshape(-1,1), x2[idx,:,1].reshape(-1,1)), 1) 
            this_jet_pt = jet_pt[idx,0]
        else:
            this_particles = [PseudoJet(px.item(), py.item(), pz.item(), e.item()) for (e, px, py, pz) in dummy_jet[:ndummy,:]]
            all_clusters.append(ClusterSequence(this_particles, jet_def))
            this_jet_particle_pts = dummy_jet_particle_pts
            this_jet_particle_etaphis = dummy_jet_particle_etaphis 
            this_jet_pt = dummy_jet_pt

        all_jet_particle_pts.append((this_jet_particle_pts/this_jet_pt).detach().cpu().numpy())
        all_jet_particle_etaphis.append(this_jet_particle_etaphis.detach().cpu().numpy())
    
    np_taus = np.zeros((len(part_ps), 3*N))
    for ii in range(N):
        taus = tauMaker(get_dR_matrix(all_clusters,
                                      all_jet_particle_etaphis,
                                      ii+1,
                                      badjets),
                        all_jet_particle_pts)
        np_taus[:, ii*3:(ii+1)*3] = taus

    tau_1_indices = [1 + i*3 for i in range(N)]
    taus = torch.from_numpy(np_taus[:, tau_1_indices]).float().to(x.device)
    taus[badjets, :] *= 0

    return taus



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
        #px_cols = [col for col in df.columns if 'PX' in col] 
        #py_cols = [col for col in df.columns if 'PY' in col]
        #pz_cols = [col for col in df.columns if 'PZ' in col]
        # df_jet_pet_eta_phi = pd.DataFrame()
        # df_jet_pet_eta_phi['jet_px'] = df[px_cols].sum(axis=1)
        # df_jet_pet_eta_phi['jet_py'] = df[py_cols].sum(axis=1)
        # df_jet_pet_eta_phi['jet_pz'] = df[pz_cols].sum(axis=1)
        #labels, prob_isQCD, prob_isSignal
        self.labels = np.expand_dims(df["is_signal_new"].to_numpy(), axis=0)
        self.labels = np.append(1-self.labels, self.labels, 0)
        
        del df
        if preprocessed:
            #jet_px = np.array(df_jet_pet_eta_phi['jet_px'])
            #jet_py = np.array(df_jet_pet_eta_phi['jet_py'])
            #jet_pz = np.array(df_jet_pet_eta_phi['jet_pz'])
            #del df_jet_pet_eta_phi
            jet_pt, jet_eta, jet_phi = get_pt_eta_phi_v(jet_px.flatten(),jet_py.flatten(),jet_pz.flatten())
            #Preprocessing
            jet_ptsum = df_pt_eta_phi[pt_cols].sum(axis=1).to_numpy().reshape(-1,1)
            df_pt_eta_phi[pt_cols]= df_pt_eta_phi[pt_cols].div(df_pt_eta_phi[pt_cols].sum(axis=1), axis=0)
            df_pt_eta_phi[eta_cols] = df_pt_eta_phi[eta_cols].subtract(pd.Series(jet_eta),axis=0)
            df_pt_eta_phi[phi_cols] = df_pt_eta_phi[phi_cols].subtract(pd.Series(jet_phi),axis=0)
            jet_pt, jet_eta, jet_phi = jet_pt.reshape(-1,1), jet_eta.reshape(-1,1), jet_phi.reshape(-1,1)
        self.aug_data = np.concatenate((jet_e, jet_m, jet_pt, jet_eta, jet_phi, jet_ptsum, jet_nconst), axis = 1)
        self.aug_data = torch.from_numpy(self.aug_data).float()
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
    for i,(x,m,l,a) in enumerate(trainloader):
        print(x.shape)
        print(m.shape)
        print(l.shape)
        print(a.shape)
        tau = processed2tau(x,a)
        print(tau.shape)
        break

