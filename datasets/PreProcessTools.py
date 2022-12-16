import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import sys, os
try:
    from fastjet import *
except:
    print("FastJet not found. Make sure FastJet has been installed and the PYTHONPATH variable includes the path to FastJet package library")
    sys.exit(1)

def h5_to_npy(file, Njets, Nstart = 0):
    jets=np.array(file.select("table",start=Nstart,stop=Nstart+Njets))
    jets2=jets[:,0:800].reshape((len(jets),200,4))
    # This way I'm getting the 1st 199 constituents.
    # jets[:,800:804] is the constituent 200.
    # jets[:,804] has a label=0 for train, 1 for test, 2 for val.
    # jets[:,805] has the label sg/bg
    labels=jets[:,805:806]
    npy_jets=[]
    for i in range(len(jets2)):
        # Get the index of non-zero entries
        nonzero_entries=jets2[i][~np.all(jets2[i] == 0, axis=1)]
        npy_jets.append([nonzero_entries,0 if labels[i] == 0 else 1])
        
    return jets2, npy_jets

def get_eta_phi(jets):
    # jets ordered as (px, py, pz, e)
    njets = jets.shape[0]
    nsubjets = jets.shape[1]
    phis = np.arctan2(jets[:,:,1], jets[:,:,0])
    etas = 0.5*np.log((jets[:,:,3] + jets[:,:,2])/(jets[:,:,3] - jets[:,:,2]))
    return np.append(phis.reshape(njets,nsubjets,1), etas.reshape(njets,nsubjets,1), axis=-1)

def get_dR_matrix(all_clusters, all_jet_particle_etaphis, Nsubjets, bad_jets):
    Njets = len(all_clusters)
    Nsubjets = int(Nsubjets)
    ## First, find the subjets from exclusive kT clustering
    all_subjets = []
    for ii, cs in enumerate(all_clusters):
        this_subjets = np.array([[jet.px(), jet.py(), jet.pz(), jet.e()] for jet in cs.exclusive_jets(Nsubjets)])
        inds = np.flip(np.argsort(this_subjets[:,3]))
        this_subjets = this_subjets[inds]
        all_subjets.append(this_subjets)

    ## Calculate subjet axes, do it for all jets
    all_subjets = np.array(all_subjets)
    # print(all_subjets.shape)
    all_subjet_axes = get_eta_phi(all_subjets)
    all_dR_matrices = []
    
    ## For all jets, calculate the dR matrix. It would be of shape (Nsubjets, Nparticles) for each jet
    for idx in range(Njets):
        this_jet_etaphis = all_jet_particle_etaphis[idx].reshape(-1, 2)
        this_jet_phis, this_jet_etas = this_jet_etaphis[:,0].reshape(1,-1), this_jet_etaphis[:,1].reshape(1,-1)
        this_jet_subjet_axes = all_subjet_axes[idx]
        this_jet_subjet_axes_phis, this_jet_subjet_axes_etas = this_jet_subjet_axes[:,0].reshape(-1,1), this_jet_subjet_axes[:,1].reshape(-1,1)
        this_dphi = this_jet_subjet_axes_phis - this_jet_phis
        this_deta = this_jet_subjet_axes_etas - this_jet_etas
        this_dR_matrix = (this_dphi**2 + this_deta**2)**0.5
        all_dR_matrices.append(this_dR_matrix)
    return all_dR_matrices



def tauMaker(all_dR_matrices, all_jet_particle_pts):
    Njets = len(all_dR_matrices)
    Nsubjets = all_dR_matrices[0].shape[0]
    taus = np.zeros((Njets, 3))
    for idx in range(Njets):
        for ib, beta in enumerate([0.5, 1 ,2]):
            minRs = np.min(all_dR_matrices[idx]**beta, axis=0)
            taus[idx, ib] = np.sum(all_jet_particle_pts[idx] * minRs)
    return taus
            
        
def prune_dataset(npy_jets, nsubjet_min = 8):
    Njets_i = len(npy_jets)
    print("Number of pre-pruned jets: ", Njets_i)
    to_remove = []
    for ii in range(Njets_i):
        jet = npy_jets[ii][0]
        if jet.shape[0] < nsubjet_min:
            print("Jet {} has {} particles. Removing it!".format(ii, jet.shape[0]))
            to_remove.append(ii)
    return to_remove



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



