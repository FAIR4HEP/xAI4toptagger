import pandas as pd
import math 
import numpy as np
import glob
import sklearn
import sys, os
from sklearn.metrics import roc_curve


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

def get_px_py_pz_v(pt, eta, phi):
    '''Provides px, py, pz, given pt, eta, and phi'''
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return px, py, pz

def rotate_v(py, pz, angle):
    '''Rotates vector by angle provided'''
    pyy = py * np.cos(angle) - pz * np.sin(angle)
    pzz = pz * np.cos(angle) + py * np.sin(angle)
    return pyy, pzz

# Train data
# Read in hdf5 files
# Note: you will need to repeat this processing for input_filenames train, test and val



input_filename = sys.argv[1]
assert os.path.exists(input_filename)
assert input_filename.endswith(".h5")
store = pd.HDFStore(input_filename)
df = store.select("table")

n_constits = 30 # use only 30 highest pt jet constituents 
df_pt_eta_phi = pd.DataFrame()

for j in range(n_constits):
    i = str(j)
    print("Processing constituent #"+str(i))
    px = np.array(df["PX_"+i][0:])
    py = np.array(df["PY_"+i][0:])
    pz = np.array(df["PZ_"+i][0:])
    pt,eta,phi = get_pt_eta_phi_v(px,py,pz)
    df_pt_eta_phi_mini = pd.DataFrame(np.stack([pt,eta,phi]).T,columns = ["pt_"+i,"eta_"+i,"phi_"+i])
    df_pt_eta_phi = pd.concat([df_pt_eta_phi,df_pt_eta_phi_mini], axis=1, sort=False)

df = df.reset_index()
df_pt_eta_phi["is_signal_new"] = df["is_signal_new"]
del df

MIN_PT = 0.0
MAX_PT = 1679.1593231 # hard-coded from paper

df_pt_eta_phi_scaled = df_pt_eta_phi.copy()
pt_cols = [col for col in df_pt_eta_phi.columns if 'pt' in col]
eta_cols = [col for col in df_pt_eta_phi.columns if 'eta' in col]
phi_cols = [col for col in df_pt_eta_phi.columns if 'phi' in col]
df_pt_eta_phi_scaled[pt_cols]= (df_pt_eta_phi.filter(regex='pt')-MIN_PT)/(MAX_PT-MIN_PT)

df_pt_eta_phi_translated = df_pt_eta_phi_scaled.copy()
df_pt_eta_phi_translated[phi_cols]= df_pt_eta_phi_scaled.filter(regex='phi')-MIN_PT

# Translate in eta by the eta of the first constituent
eta_shift = df_pt_eta_phi_scaled['eta_0']
df_pt_eta_phi_translated[eta_cols] = df_pt_eta_phi_scaled.filter(regex='eta')-np.tile(eta_shift, [n_constits,1]).T

# Translate in phi 
phi_shift = np.array(df_pt_eta_phi_scaled['phi_0'])
x = np.where(phi_shift < -math.pi)
y = np.where(phi_shift >= math.pi)
if x:
    phi_shift[x] = phi_shift[x]+2*math.pi
if y:
    phi_shift[y] = phi_shift[y]-2*math.pi

df_pt_eta_phi_translated[phi_cols] = df_pt_eta_phi_scaled.filter(regex='phi')-np.tile(phi_shift, [n_constits,1]).T
del df_pt_eta_phi_scaled

df_pt_eta_phi_rotated = pd.DataFrame()

# Calculate theta from second jet constituents
pt =  np.array(df_pt_eta_phi_translated['pt_1'])[0:]
eta = np.array(df_pt_eta_phi_translated['eta_1'])[0:]
phi = np.array(df_pt_eta_phi_translated['phi_1'])[0:]
px, py, pz = get_px_py_pz_v(pt,eta,phi)
theta = np.arctan2(py, pz) + math.pi/2

for j in range(n_constits):
    i = str(j)
    pt =  np.array(df_pt_eta_phi_translated['pt_'+i])[0:]
    eta = np.array(df_pt_eta_phi_translated['eta_'+i])[0:]
    phi = np.array(df_pt_eta_phi_translated['phi_'+i])[0:]
     
    # Rotate by theta
    px, py, pz = get_px_py_pz_v(pt,eta,phi)
    py, pz = rotate_v(py, pz, theta)
    pt, eta, phi = get_pt_eta_phi_v(px, py, pz)
    df_pt_eta_phi_mini = pd.DataFrame(np.stack([pt,eta,phi]).T,columns = ["pt_"+i,"eta_"+i,"phi_"+i])
    df_pt_eta_phi_rotated = pd.concat([df_pt_eta_phi_rotated,df_pt_eta_phi_mini], axis=1, sort=False)

df_pt_eta_phi_rotated["is_signal_new"] = df_pt_eta_phi_translated["is_signal_new"]
del df_pt_eta_phi_translated

# Move average jet pt to right hand plane
eta_times_pt = np.multiply(df_pt_eta_phi_rotated.filter(regex='pt'),df_pt_eta_phi_rotated.filter(regex='eta'))
centre = np.sum(eta_times_pt,axis=1)
x = np.where(centre < 0)
df_pt_eta_phi_flipped = df_pt_eta_phi_rotated.copy()
for col in eta_cols: # funny syntax. 
    df_pt_eta_phi_flipped[col].loc[x] = -1.0*df_pt_eta_phi_flipped[col].loc[x]

del df_pt_eta_phi_rotated

df_pt_eta_phi_flipped.to_pickle("./topoprocessed/" + input_filename.replace(".h5", "") + ".pkl")
