#%%
# Fetch data from reproducibility.org
#sf.Fetch('marm','marmvel.hh')

# #%% [markdown]
#    ## Generating the model
#    First we create model by augmenting Marmousi II
#%% [markdown]
# ## Generator features
# 
# ### Several layers can be generated from a single one
# 
# ### Velocities are exactly the same as in mother-model (Marmousi)
# 
#%%
#(c) Vladimir Kazei, Oleg Ovcharenko; KAUST 2020
# cell with imports
import importlib
import multiprocessing
import os
import os.path
import sys
import time
import pickle
import threading
import random
from numba import jit

# madagascar API
import m8r as sf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as matlib
import pydot
#plt.rcParams.update({'font.size': 5002})
#plt.rcParams['figure.figsize'] = [10, 5]
import seaborn
import tensorflow as tf


# images
from IPython import get_ipython
from keras import backend as K
from keras.utils import multi_gpu_model

from numpy.random import randint, seed
from scipy import ndimage
from skimage.transform import resize

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
#import styler
from myutils import (cd, cmd, const, elastic_transform, plt_nb_T, toc, aug_flip, 
                     merge_dict, np_to_rsf, rsf_to_np, nrms, tf_random_flip_channels)
from myutils import const as c


seed()
# set up matplotlib
matplotlib.rc('image', cmap='RdBu_r')
seaborn.set_context('paper', font_scale=5)

CUDA_VISIBLE_DEVICES = "1,2,3"

os.environ["CUDA_VISIBLE_DEVICES"]=CUDA_VISIBLE_DEVICES
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Madagascar binaries will be stored in DATAPATH (RAM on Linux recommended)
cmd("mkdir /dev/shm/RSFTMP")
cmd("chmod 777 /dev/shm/RSFTMP")
os.environ["DATAPATH"]="/dev/shm/RSFTMP/"


#%%

alpha_deform = 500
sigma_deform = 50

def generate_model(model_input=c.trmodel, 
                   model_output="marm.rsf",
                   dx=c.dx,
                   stretch_X=1,
                   training_flag=False,
                   random_state_number=c.random_state_number,                  
                   distort_flag=True,
                   crop_flag=True,
                   verbose=False,
                   test_flag=False,
                   show_flag=False):
    # downscale marmousi
    #def rescale_to_dx(rsf_file_in, rsf_file_out, dx)
    model_orig = sf.Input(model_input)
    vel = model_orig.read()
    
    if test_flag:
        n_cut = int(((const.sxbeg + const.gxbeg) * const.dx) // (model_orig.float("d1")))
        vel = np.concatenate((vel[-n_cut:,:], np.flipud(vel), vel[:n_cut,:]), axis = 0)
    else:
        vel = np.concatenate((vel, np.flipud(vel), vel), axis = 0)
        
    if show_flag:
        np.random.RandomState(random_state_number)
        random.seed(random_state_number)
        np.random.seed(random_state_number)
        
    if crop_flag:
        vel_log_res = vel
        #vel_log_res = resize(vel_log_res[:,:], (np.shape(vel)[0]//2, np.shape(vel)[1]//2))
        if verbose:
            print(f"Random state number = {random_state_number}")
        #vel = resize(vel_log_res, vel.shape)
        
        l0 = randint(np.shape(vel)[0])
        #print(f"l0={l0}")
       
        h0 = min(l0 + np.shape(vel)[0]//4 + randint(np.shape(vel)[0]//2), 
                 np.shape(vel)[0])
        l1 = randint(np.shape(vel)[1]//3)
        h1 = min(l1 + np.shape(vel)[1]//3 + randint(np.shape(vel)[1]//2), 
                 np.shape(vel)[1])
        if verbose:
            print(l0, l1, h0, h1)
        vel_log_res = vel_log_res[l0:h0, l1:h1]
        
        vel = resize(vel_log_res, vel.shape)
    # we downscale
    scale_factor = dx / model_orig.float("d1")
    
    
    vel = resize(vel[:,:], (stretch_X*np.shape(vel)[0]//scale_factor, np.shape(vel)[1]//scale_factor))
    
    if verbose:
        print(np.shape(vel))
        print(f"Model downscaled {scale_factor} times to {dx} meter sampling")
        if stretch_X != 1:
            print(f"Model stretched {stretch_X} times to {dx} meter sampling \n")
    
    # we concatenate horizontally, this is confusing because of flipped axis in madagascar
    
    vel = np.atleast_3d(vel)
    
    if distort_flag:
        vel = elastic_transform(vel, alpha_deform, sigma_deform, v_dx=dx, random_state_number=random_state_number)
    vel = np.squeeze(vel)
    
    if distort_flag:
        vel_alpha = (0.8+0.4*resize(np.random.rand(5,10), vel.shape))
        #print(vel_alpha)
        vel *= vel_alpha
    # add water
    # vel = np.concatenate((1500*np.ones((vel.shape[0], 20)), vel), 
    #                      axis=1)
    #vel = ndimage.median_filter(vel, size=(7,3))
    #vel = 1500 * np.ones_like(vel)
    if verbose:
        print(f"Writing to {model_output}")
        
    np_to_rsf(vel, model_output)
    return vel


def show_model_generation():
    stretch_X_train = c.stretch_X_train
    vel = generate_model(stretch_X=stretch_X_train, training_flag=False, crop_flag=False, distort_flag=False, random_state_number=randint(10000))
    
    vel = rsf_to_np("marmvel.hh")
    plt_nb_T(aug_flip(vel), dx=4, dz=4, fname="../latex/Fig/marm_aug")

    vel = generate_model(stretch_X=stretch_X_train, distort_flag=False, random_state_number=c.random_state_number, show_flag=True)
    plt_nb_T(vel, fname="../latex/Fig/cropMarm")
    N = np.shape(vel)

    vel_example = elastic_transform(np.atleast_3d(vel), alpha_deform, sigma_deform, 
                                    random_state_number=c.random_state_number, plot_name="Normal")

    N = np.shape(vel)
    vel_example = generate_model(stretch_X=stretch_X_train, training_flag=True, random_state_number=c.random_state_number, show_flag=True)
    vel1 = generate_model(stretch_X=stretch_X_train, training_flag=False, random_state_number=randint(10000))
    vel2 = generate_model(stretch_X=stretch_X_train, training_flag=False, random_state_number=randint(10000))
    vel3 = generate_model(stretch_X=stretch_X_train, training_flag=False, random_state_number=randint(10000))
    vel4 = generate_model(stretch_X=stretch_X_train, training_flag=False, random_state_number=randint(10000))
    plt_nb_T(np.concatenate((vel_example, vel1, vel2, vel3, vel4), axis=1), fname="../latex/Fig/random_model_example")

# model data and sort into CMPs function
def generate_rsf_data(model_name="marm.rsf", central_freq=c.central_freq, dt=c.dt, dx=c.dx, 
                        nt=c.nt, sxbeg=c.sxbeg, gxbeg=c.gxbeg, szbeg=c.szbeg, 
                        jsx=c.jsx, jgx=c.jgx, jdt=c.jdt,
                        logs_out="logs.rsf", shots_out="shots_cmp.rsf",
                        full_shots_out=None): 
    
    #get size of the model
    model_orig = sf.Input(model_name)
    Nx = model_orig.int("n2")
    print(Nx)
    ns = (Nx - 2*sxbeg)//jgx 
    ng = 2*(sxbeg-gxbeg)//jgx + 1
    print(f"Total number of shots = {ns}")
    t_start = time.time()
    cmd((f"sfgenshots < {model_name} csdgather=y fm={central_freq} amp=1 dt={dt} ns={ns} ng={ng} nt={nt} "
         f"sxbeg={sxbeg} chk=n szbeg={szbeg} jsx={jgx} jsz=0 gxbeg={gxbeg} gzbeg={szbeg} jgx={jgx} jgz=0 > shots.rsf"))
    print(f"Modeling time for {ns} shots = {time.time()-t_start}")
    if full_shots_out != None:
        cmd(f"sfcp < shots.rsf > {full_shots_out}")

    #   ## Analyze and filter the data set generated
    # correct header and reduce sampling in time jdt (usually 4) times
    cmd(f"sfput < shots.rsf d3={jgx*dx} | sfwindow j1={jdt} | sfbandpass flo=2 fhi=4 > shots_decimated.rsf")
    cmd(f"sfrm shots.rsf")
    # sort into cmp gathers and discard odd cmps and not full cmps
    cmd(f"sfshot2cmp < shots_decimated.rsf half=n | sfwindow j3=2 f3={ng//2} n3={ns} > {shots_out}")
    print(f"sfshot2cmp < shots_decimated.rsf half=n | sfwindow j3=2 f3={ng//2} n3={ns} > {shots_out}")
    # cmd(f"sfrm shots_decimated.rsf")
    # cmd(f"sfrm shots_decimated.rsf")
    # create the logs -- training outputs
    cmd(f"sfsmooth < {model_name} rect2=2 | sfwindow f2={sxbeg} j2={jsx} n2={ns} > {logs_out}")
    #cmd(f"sfin < {logs_out}")
    return 0

#@profile
def generate_rsf_data_multi(model_name="marm.rsf", central_freq=c.central_freq, dt=c.dt, 
                        nt=c.nt, sxbeg=c.sxbeg, gxbeg=c.gxbeg, szbeg=c.szbeg, 
                        jsx=c.jsx, jgx=c.jgx, jdt=c.jdt,
                        logs_out="logs.rsf", shots_out="shots_cmp.rsf", iShotBlock=None): 
    cmd(f"mkdir /dev/shm/RSFTMP/data_{iShotBlock}")
    cmd(f"chmod 777 /dev/shm/RSFTMP/data_{iShotBlock}")    
    os.environ["DATAPATH"]=f"/dev/shm/RSFTMP/data_{iShotBlock}"
    cmd(f"echo $DATAPATH")
    cmd(f"mkdir data_{iShotBlock}")
    seed()
    #cmd(f"sfwindow < overthrust3D.hh n3=120 f1={iShotBlock*randint(0,1e7) % 400} n1=1 | sftransp | sfadd scale=1000 | sfput d1=25 d2=25 --out=stdout > data_{iShotBlock}/overthrust2D.hh")
    #cmd(f"cp {c.trmodel} data_{iShotBlock}/")
    with cd(f"data_{iShotBlock}"):
        _vel = generate_model(model_input=f"../{c.trmodel}", random_state_number=(iShotBlock + randint(0,1e7)))
        #plt_nb_T(_vel)
        generate_rsf_data()


##%%
def generate_all_data(random_model_repeat=c.random_model_repeat):

    K.clear_session()
    start_modeling_time = time.time()
    procs = []
    for iShotBlock in range(random_model_repeat):
        # we run modeling on 1-3 GPUs, GPU 0 is for the network
        os.environ["CUDA_VISIBLE_DEVICES"] = str((iShotBlock % 3) + 1)
        proc = multiprocessing.Process(target=generate_rsf_data_multi, kwargs ={'iShotBlock' : iShotBlock})
        proc.start()
        procs.append(proc)
        if len(procs) > 100:
            for proc in procs[:50]:
                proc.join()
            procs = procs[50:]
    for proc in procs:
        proc.join()
    print(f"Time for modeling = {toc(start_modeling_time)}")
    
    start_merging_rsf_time = time.time()        
    cmd(f"sfcat data_*/shots_cmp.rsf axis=4 > shots_cmp_full.rsf")
    cmd(f"sfcat data_*/logs.rsf axis=3 > logs_full.rsf")
    
    print(f"Time for merging rsf files = {toc(start_merging_rsf_time)}")
    os.environ["CUDA_VISIBLE_DEVICES"]=CUDA_VISIBLE_DEVICES

# generate_all_data()  

if __name__ == "__main__":
    time_start = time.time()
    cmd("rm new_data_ready")
    print("LOOOOOOOOOOOOOOOOP for data generation started")
    sleep_counter = 0
    while True:
        if  os.path.exists('training_finished'):
            cmd("rm training_finished")
            print("data generation finished as training_finished")
            print(f"total execution time = {time.time()-time_start}, total idle = {sleep_counter}s")
            quit()
        elif os.path.exists('new_data_ready'):
            sleep_counter += 1
            print(f"network is not ready for new data, sleep {sleep_counter} sec", end='\r')
            time.sleep(1)
        else:
            generate_all_data()
            cmd("touch new_data_ready")


