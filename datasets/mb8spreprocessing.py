import pandas as pd
import numpy as np
import sys, os
from fastjet import *
from PreProcessTools import *


if __name__ == "__main__":
    data_loc = "./"
    data_dest = "./n-subjettiness_data/"
    if not os.path.exists(data_dest):
        os.mkdir(data_dest)
    data_files = ['train.h5', 'val.h5',  'test.h5']
    Njets = None
    Njets_per_file = 100000
    Nsubjets = list(range(1,9))
    jet_def = JetDefinition(kt_algorithm, 0.4)

    for filename in data_files:
        print("Filename: ", filename)
        file = pd.HDFStore(data_loc+filename)
        Njets_total = file.get_storer('table').nrows
        nfiles = int(Njets_total/Njets_per_file)
        for fid in range(nfiles):
            print("File index: ", fid)
            _, npy_jets = h5_to_npy(file, Njets=Njets_per_file, Nstart=fid*Njets_per_file)
            bad_jets = prune_dataset(npy_jets, nsubjet_min = 8)
            Njets = len(npy_jets) - len(bad_jets)
            print("Number of jets after pruning:", Njets)
            all_clusters = []
            all_jet_particle_pts = []
            all_jet_particle_etaphis = []
            all_labels = []
            all_jet_pts = []
            all_jet_masses = []
            for idx, (jet, label) in enumerate(npy_jets):
                if idx in bad_jets:
                    continue
                # print("Jet tag: ", label)
                all_labels.append(label)
                this_particles = [PseudoJet(px, py, pz, e) for (e, px, py, pz) in jet]
                all_clusters.append(ClusterSequence(this_particles, jet_def))
                this_jet_particle_pts = ((jet[:,1]**2 + jet[:,2]**2)**0.5).reshape(-1)
                this_jet_particle_etaphis = get_eta_phi(jet[:,[1,2,3,0]].reshape(1,-1,4)).reshape(-1,2)
                this_jet_pt = (np.sum(jet[:,1])**2 + np.sum(jet[:,2])**2)**0.5
                this_jet_mass = np.abs(np.sum(jet[:,0])**2 - this_jet_pt**2 - np.sum(jet[:,3])**2)**0.5
                all_jet_particle_pts.append(this_jet_particle_pts/np.sum(this_jet_pt))
                all_jet_particle_etaphis.append(this_jet_particle_etaphis)
                all_jet_pts.append(this_jet_pt)
                all_jet_masses.append(this_jet_mass)
            all_labels = np.array(all_labels)
            all_jet_pts = np.array(all_jet_pts)
            all_jet_masses = np.array(all_jet_masses)
            np_taus = np.zeros((Njets, 3*len(Nsubjets)))
            for ii, nsubjet in enumerate(Nsubjets):
                taus = tauMaker(get_dR_matrix(all_clusters,
                                              all_jet_particle_etaphis,
                                              nsubjet,
                                              bad_jets),
                                all_jet_particle_pts)
                np_taus[:, ii*3:(ii+1)*3] = taus
            np_taus = np.concatenate((np_taus, all_jet_pts.reshape(-1,1), all_jet_masses.reshape(-1,1), all_labels.reshape(-1,1)), axis=1)  
            print("Dataset size: ", np_taus.shape)
            np.save(data_dest + filename.replace(".h5", "_{}.npy".format(fid)), np_taus)
        print("{} has been converted".format(filename))
        file.close()

        dtype = filename.replace(".h5", "")
        files = [f for f in os.listdir(data_dest) if dtype in f and "all" not in f]
        print("Number of {} files: {}".format(dtype, len(files)))
        for ii, f in enumerate(files):
            if ii == 0 :
                all_data = np.load(data_dest + f)
            else:
                all_data = np.append(all_data, np.load(data_dest + f), axis=0)
        np.save(data_dest + dtype + "_all.npy", all_data)
        for f in files:
            os.remove(data_dest + f)
        print(dtype, "data is merged!")
