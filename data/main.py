#%% [markdown]
#  # Smart velocity analysis : mapping raw data to velocity logs
#%%
#(c) Vladimir Kazei, Oleg Ovcharenko; KAUST 2020
# cell with imports
import importlib
import multiprocessing
import os
import sys
import time
import pickle
import threading
import random

# learning
import keras
# madagascar API
import m8r as sf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import tensorflow as tf


# images
from IPython import get_ipython
from keras import backend as K
from keras.utils import multi_gpu_model

from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.layers import (AveragePooling2D, BatchNormalization, Conv2D, Dense, Lambda,
                          Dropout, Flatten, MaxPool2D, Reshape, GaussianNoise, GaussianDropout)
from keras.models import load_model
from numpy.random import randint, seed
from scipy import ndimage
from skimage.transform import resize
from skimage.util import view_as_windows

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#import styler
from myutils import (cd, cmd, const, 
                     elastic_transform, plt_nb_T, toc, aug_flip, upsample,
                     merge_dict, np_to_rsf, rsf_to_np, nrms, 
                     tf_random_flip_channels)

from myutils import const as c

from generate_data import (generate_model, show_model_generation, 
                           alpha_deform, sigma_deform, 
                           generate_all_data, generate_rsf_data)

seed()
# set up matplotlib
matplotlib.rc('image', cmap='RdBu_r')
seaborn.set_context('paper', font_scale=5)

CUDA_VISIBLE_DEVICES = "0"
os.environ["CUDA_VISIBLE_DEVICES"]=CUDA_VISIBLE_DEVICES
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Madagascar binaries will be stored in DATAPATH (RAM on Linux recommended)
cmd("mkdir /dev/shm/RSFTMP")
cmd("chmod 777 /dev/shm/RSFTMP")
os.environ["DATAPATH"]="/dev/shm/RSFTMP/"

# execution flags
generate_rsf_data_flag = True
retrain_flag = False #(sys.argv[1] == "--retrain")
print(f"retrain_flag = {retrain_flag}")
print(type(retrain_flag))

tic_total = time.time()
#%% [markdown]
# ## Introduction
# 
# ### Why?
# FWI provides high resolution models, yet it is very computationally expensive and it can fail with the lack of low frequencies. 
# 
# Velocity analysis is on the other hand very cheap computationally, but limited by the assumptions on the background medium.
# 
# ### Goal
# Combine advantages of both methods with deep learning
# 
# ### Solution
# We will train a deep convolutional neural network to perform velocity analysis in inhomogeneous media
##%% [markdown]
# We estimate velocity $v(x_{CMP}, z)$ from presure field
#  $p_{obs}(x_{CMP}-\varepsilon:x_{CMP}+\varepsilon, 0:h_{max}, f)$, where
#  $x_{CMP}$ is the central midpoint,
#  $p_{obs}$ is the observed pressure.
#  
# $\varepsilon = 0$ in this first part of the application => single CMP as input
##%% [markdown]
# ## Method
# 
# 0) generate a model set
# 1) generate seismic data set
# 2) build neural network
# 3) train neural network
# 4) test it on a model that it has not seen
    

#%% [markdown]
# ## Model generation
# 
# we utilize common deep learning image augmentation technique -- elastic transform

#%%
show_model_generation()

#%% [markdown]
# ## Gaussian fields to generate a coordinate shift for laterally smooth models
# 
# 
# 
# ### Large correlation radius in horizontal direction -- to keep it almost horizontally layered
# 
# ### Small correlation radius in vertical direction -- to make it represent different layering scenarios
# 
# ### Same parameters but different fields for horizontal and vertical components
# 
# ### Large vertical shifts and small horizontal -- to keep it laterally slowly varying



#%% [markdown]
# ## Modeling data with constant offset on GPU with Madagascar

#%%
# Setting up parameters
_vel = generate_model()
N = np.shape(_vel)
dt = c.dt
dx = c.dx
T_max = c.T_max
nt = c.nt
print(f"number of time steps = {nt}")
# check stability
print(f"you chose dt = {dt}, dt < {dx/np.max(_vel):.4f} should be chosen for stability \n")
# force stability
assert dt < dx/np.max(_vel)

# ricker wavelet is roughly bounded by 3f_dominant
# therefore the sampling rate principally acceptable sampling rate would be
central_freq = c.central_freq
print(f"dt from Nyquist criterion is {1/(2*3*central_freq)}")
print(f"dt chosen for CNN is {c.jdt*dt}, which is {(1/(3*central_freq))/(c.jdt*dt)} samples per cycle")

#%% [markdown]
# ## Read data into numpy and check that the number of logs is the same as number of shots

#%%
nCMP=21
def read_rsf_XT(shots_rsf='shots_cmp_full.rsf', logs_rsf='logs_full.rsf', j_log_z=c.jlogz):
    
    X = rsf_to_np(shots_rsf)
    # single model exception
    if X.ndim == 3:
        X = np.expand_dims(X, axis=0)    
    
    X_f = np.flip(X, axis=2)
    X = np.maximum(np.abs(X), np.abs(X_f)) * np.sign(X+X_f)
    X = X[:,:,:(np.shape(X)[2] + 1) // 2,:]
    
    T = rsf_to_np(logs_rsf)
    # single model exception
    if T.ndim == 2:
        T = np.expand_dims(T, axis=0)
    # decimate logs in vertical direction --2 times by default
    T = resize(T, (*T.shape[0:2], np.shape(T)[2] // j_log_z))
    T_size = np.shape(T)
    print(T_size)

    # ensure that the number of logs is equal to the number of CMPs
    assert (X.shape[0:2] == T.shape[0:2])
    return X, T

#%%
while not os.path.exists('new_data_ready'):
    time.sleep(1)
    print("waiting for new data, run python generate_data.py if you didn't", end="\r")
cmd("rm new_data_ready")

#%%
X, T = read_rsf_XT(shots_rsf='shots_cmp_full.rsf', logs_rsf='logs_full.rsf')

T_multi = view_as_windows(T, (1, nCMP, T.shape[2])).squeeze().reshape((-1, nCMP, T.shape[2]))[:,nCMP//2,:].squeeze()

# create scaler for the outputs
T_scaler = StandardScaler().fit(T_multi)
scale = np.copy(T_scaler.scale_)
mean = np.copy(T_scaler.mean_)
np.save("scale", scale)
np.save("mean", mean)

#%%
T_scaler.scale_[:] = 1
T_scaler.mean_[:] = 0

# X has the format (model, CMP, offset, time)

plt_nb_T(X[1,:10, -1,:200], title="Common offset (600 m) gather", dx=c.dx*c.jgx*2, dz=1e3*dt*c.jdt, 
        origin_in_middle=True, ylabel="Time(s)", fname="../latex/Fig/X_short_offset", vmin=-1e-4, vmax=1e-4)

plt_nb_T(T[1,:10,:100], title="Model", dx=c.dx*c.jgx*2, dz=c.dx*c.jlogz, 
        origin_in_middle=True, ylabel="Time(s)", fname="../latex/Fig/X_short_offset")

#X=X[:,:-3,:,:]

#%%
def prepare_XT(X,T, T_scaler=T_scaler, gen_plots=False):
    nCMP = 21
    X_multi = view_as_windows(X, (1,nCMP,X.shape[2],X.shape[3])).squeeze().reshape((-1, nCMP, X.shape[2], X.shape[3]))
    X_multi = np.swapaxes(X_multi,1,3)
    X_multi = np.swapaxes(X_multi,1,2)
    
    T_multi = view_as_windows(T, (1, nCMP, T.shape[2])).squeeze().reshape((-1, nCMP, T.shape[2]))[:,nCMP//2,:].squeeze()
    

    X_scaled_multi = X_multi
    T_scaled_multi = T_scaler.transform(T_multi)
    # extract central CMPs for singleCMP network
    X_scaled = X_scaled_multi[:,:,:,nCMP//2:nCMP//2+1]
    T_scaled = T_scaled_multi
    #%%
    if gen_plots:
        plt_nb_T(T_multi, dx=c.jgx*c.dx, dz=c.jlogz*c.dx, fname="../latex/Fig/T_multi")
        plt_nb_T(1e3*T_scaled, dx=c.jgx*dx, dz=c.jlogz*c.dx, fname="../latex/Fig/T_scaled")

        #%%
        # show single training sample
        sample_reveal = nCMP
        plt_nb_T(1e3*np.concatenate((np.squeeze(X_scaled_multi[sample_reveal,:,:,-1]), np.flipud(np.squeeze(X_scaled_multi[sample_reveal,:,:,0]))), axis=0),
                title="CMP first | CMP last", dx=200, dz=1e3*dt*c.jdt, 
                origin_in_middle=True, ylabel="Time(s)", fname="../latex/Fig/X_scaled", cbar_label = "")
        print(np.shape(1e3*T_scaled[sample_reveal-(nCMP+1)//2:sample_reveal+(nCMP-1)//2:nCMP]))

        plt_nb_T(1e3*T_multi[sample_reveal-(nCMP-1)//2:sample_reveal+(nCMP-1)//2,:], 
                dx=100, dz=c.dx*c.jlogz, 
                title="scaled velocity logs")
    
    return X_scaled_multi, T_scaled_multi

X_scaled_multi, T_scaled_multi = prepare_XT(X,T, T_scaler=T_scaler, gen_plots=False)

#%% plot single input into the network
plt_nb_T(1e3*np.reshape(X[0, :21, :, :], (21*X.shape[2], X.shape[3])), vmin=-0.1, vmax=0.1, figsize=(48,12), no_labels=True, cbar=True, fname="../latex/Fig/input_multi")


#%% [markdown]
#    # CNN construction single CMP -> log under the CMP

# 1D total variation for the output
def tv_loss(y_true, y_pred):
    #b, h, w, c = img.shape.as_list()
    a = K.abs(y_pred[:, :-1] - y_pred[:, 1:])
    tv = 0.0 * K.mean(a, axis=-1)
    total = tv + K.mean(K.square(y_pred - y_true), axis=-1)
    return total

def random_channel_flip(x):
    print(x.shape)
    return K.in_train_phase(tf_random_flip_channels(x), x)

def R2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true-y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def create_model(inp_shape, out_shape, jlogz=c.jlogz):
    model = keras.models.Sequential()
    activation = 'elu'
    padding = 'same'
    kernel_size = (3, 11)
    model.add(Lambda(random_channel_flip, input_shape=inp_shape, output_shape=inp_shape))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(filters=32, kernel_size=kernel_size, activation=activation, padding=padding, input_shape=inp_shape))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(filters=32, kernel_size=kernel_size, strides=(2,2), activation=activation, padding=padding))       
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(filters=64, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(filters=64, kernel_size=kernel_size, strides=(2,2), activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(filters=128, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(filters=128, kernel_size=kernel_size, strides=(2,2), activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(filters=64, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(filters=64, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(filters=32, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(filters=32, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(filters=16, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(filters=16, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(filters=8, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(filters=8, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(filters=4, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(filters=4, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(filters=2, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(filters=2, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(filters=1, kernel_size=(3, 15), activation='linear', padding="valid"))
    model.add(Flatten())
    model.add(Lambda(lambda x: K.tf.add(K.tf.multiply(x, K.variable(scale.squeeze)), 
                                        K.variable(mean.squeeze))))
    return model


#%%

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def batch_generator(X, T, T_scaler=T_scaler, batch_size = None):
    batch=[]
    print("generator restarted !!!!!!!!!!!!!!!!!!!!!!!!!!! waiting for new data")
    while not os.path.exists("new_data_ready"):
        time.sleep(1)    
    while True:
        # it might be a good idea to shuffle your data before each epoch
        # for iData in range(40):
        #     print(f"loading NEW DATA {iData}")
        #     X_rsf, T_rsf = read_rsf_XT(shots_rsf=f'/data/ibex_data/fullCMP_{iData}/shots_cmp_full.hh', 
        #                                logs_rsf=f'/data/ibex_data/fullCMP_{iData}/logs_cmp_full.hh')
        #     X, T = prepare_XT(X_rsf, T_rsf, T_scaler)
        
        #     indices = np.arange(len(X))
        #     np.random.shuffle(indices)
        #     for i in indices:
        #         # if os.path.exists("new_data_ready"):
        #         #     break
        #         batch.append(i)
        #         if len(batch)==batch_size:
        #             yield X[batch], T[batch]
        #             batch=[]
        if os.path.exists("new_data_ready"):
            cmd("rm new_data_ready")
            X_rsf, T_rsf = read_rsf_XT(shots_rsf='shots_cmp_full.rsf', logs_rsf='logs_full.rsf')
            #cmd("ssh glogin.ibex.kaust.edu.sa 'rm ~/log_estimation/data/new_data_ready'")
            X, T = prepare_XT(X_rsf, T_rsf, T_scaler)
            print("new data loaded")
        else:
            print("reusing the old data")
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        print("indices reshuffled") 
        for i in indices:
            if os.path.exists("new_data_ready"):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
                break
            batch.append(i)
            if len(batch)==batch_size:
                yield X[batch], T[batch]
                batch=[]

#%%
# Init callbacks
def train_model(prefix="multi", X_scaled=X_scaled_multi, T_scaled=T_scaled_multi, weights=None):
    cmd("rm new_data_ready")
    #cmd("ssh 10.109.66.7 'rm ~/log_estimation/data/new_data_ready'")
    lr_start = 0.001
    if weights != None:
        lr_start = 1e-5
    net = create_model(np.shape(X_scaled)[1:], np.shape(T_scaled)[1:]) 
    net.compile(loss=tv_loss,
                  optimizer=keras.optimizers.Nadam(lr_start),
                  metrics=[R2])
    
    #net.summary()
    if weights != None:
        net.load_weights(weights)
        
    early_stopping = EarlyStopping(monitor='val_loss', patience=21)
    model_checkpoint = ModelCheckpoint("trained_net",
                                       monitor='val_loss',
                                       save_best_only=True,
                                       verbose=1,
                                       period=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=7, min_lr=1e-5, verbose=1)

    X_valid = X_scaled
    T_valid = T_scaled
    print(f"X validation data size = {np.shape(X_valid)}")
    
    # TRAINING
    batch_size = 64
    # we flip every batch, so, going through whole data needs twice as many batches
    steps_per_epoch = len(X_scaled)//batch_size
    print(f"Batch size = {batch_size}, batches per epoch = {steps_per_epoch}")
    history = net.fit_generator(batch_generator(X_scaled, T_scaled, batch_size=batch_size),
                                validation_data=(X_valid, T_valid),
                                epochs=200,
                                verbose=2,
                                shuffle=True,
                                max_queue_size=200,
                                workers=10, 
                                use_multiprocessing=False,
                                steps_per_epoch = steps_per_epoch,
                                callbacks=[model_checkpoint,
                                reduce_lr,
                                early_stopping])   

    
    print("Optimization Finished!")
        
    return net, history

def load_history(fname_history):
    with open(fname_history,'rb') as f:
        return pickle.load(f)

def save_history(history, fname_history):
    with open(fname_history,'wb') as f:
        pickle.dump(history, f)

def train_ensemble(prefix, X_scaled, T_scaled):
    valid_best=1e100
    net_dict = {}
    history_dict = {}
    for iNet in range(5):
        if retrain_flag:
            weights = f"{prefix}_weights.h5"
            history_prev = load_history(f"history_{prefix}")   
        else:
            weights = None
        
        net, history = train_model(prefix=prefix, X_scaled=X_scaled, T_scaled=T_scaled, weights=weights)
        cur_val_loss = np.min(history.history['val_loss'])
        print(cur_val_loss)
        if  cur_val_loss < valid_best:
            valid_best =  cur_val_loss
            net_best = net
            history_best = history.history        
        net_dict[f"{iNet}"] = net
        history_dict[f"{iNet}"] = history.history
        if retrain_flag:
            history_best = merge_dict(history_prev, history_best)
        
        net_best.save_weights(f"{prefix}_weights.h5")
        save_history(history_best, f"history_{prefix}")
        
    return net_dict, history_dict, net_best, history_best

#singleCMP_net_dict, singleCMP_net_best, history_best = train_ensemble("singleCMP", X_scaled, T_scaled)
#cmd("rm new_data_ready")
multiCMP_net_dict, history_dict, multiCMP_net_best, history_best = train_ensemble("multiCMP", X_scaled_multi, T_scaled_multi)
# stop generator
cmd("touch training_finished")
#%% KOSTYLI for testing
history_best = load_history("history_multiCMP")
prefix = "multiCMP"
plt.figure(figsize=(16,9))
r2_arr = np.zeros((history_dict.__len__(),2))
for iNet in history_dict.keys():
    r2_arr[int(iNet),0] = history_dict[iNet]['R2'][-1]
    r2_arr[int(iNet),1] = history_dict[iNet]['val_R2'][-1]
    print(f"netN={iNet}, R2={history_dict[iNet]['R2'][-1]},{history_dict[iNet]['val_R2'][-1]}")
    plt.plot(history_dict[iNet]['R2'][:],'b--')
    plt.plot(history_dict[iNet]['val_R2'][:],'r')

print(f"Average R2={np.mean(r2_arr, 0)}")

plt.plot(history_best['R2'][:],'b--', label='Training R2', linewidth=3)
plt.plot(history_best['val_R2'][:],'r', label='Validation R2', linewidth=3)
plt.xlabel("epoch")
plt.legend()    
plt.savefig(f"../latex/Fig/{prefix}_R2", bbox_inches='tight')
plt.grid(True,which="both",ls="-")
plt.show(block=False)
plt.pause(1)
plt.close()

plt.figure(figsize=(16,9))
for iNet in history_dict.keys():
    print(iNet)
    plt.plot(history_dict[iNet]['loss'][:],'b--')
    plt.plot(history_dict[iNet]['val_loss'][:],'r')

plt.semilogy(history_best['loss'][:],'b--', label='Training loss', linewidth=3)
plt.semilogy(history_best['val_loss'][:],'r', label='Validation loss', linewidth=3)
plt.xlabel("epoch")
plt.legend()    
plt.savefig(f"../latex/Fig/{prefix}_loss", bbox_inches='tight')
plt.grid(True,which="both",ls="-")
plt.show(block=False)
plt.pause(1)
plt.close()


# #%%
# multiCMP_net_dict={}
# net_best = create_model(np.shape(X_scaled_multi)[1:], np.shape(T_scaled_multi)[1:])
# net_best.summary()
# net_best.compile(loss=tv_loss,
#                  optimizer=keras.optimizers.Nadam(1e-6),
#                  metrics=[R2])
# net_best.load_weights("multiCMP_weights.h5")
# multiCMP_net_dict["0"] = net_best




#%% [markdown]
# # We trained the neural net, it fits the training and validation data... 
# 
# ## How well does it fit?
# 
# ## Does it fit stretched marmousi itself?
# 
# ## Could we learn more from models like this?
# 
# ## Does it work on something different?
# 
# ## When does it break?!
#%% [markdown]
# # Testing
#%% uncomment for loading initial weights
# singleCMP_net_dict={}
# net = create_model(np.shape(X_scaled)[1:], np.shape(T_scaled)[1:])
# net.summary()
# net.load_weights("singleCMP_weights.h5")
# singleCMP_net_dict["0"] = net

# multiCMP_net_dict={}
# netM = create_model(np.shape(X_scaled_multi)[1:], np.shape(T_scaled_multi)[1:])
# netM.summary()
# netM.load_weights("multiCMP_weights.h5")
# multiCMP_net_dict["0"] = netM



def test_on_model(folder="marmvel1D",
                  net_dict=None,
                  prefix="singleCMP",
                  model_filename=None, 
                  distort_flag=False,
                  stretch_X=None,
                  nCMP_max=nCMP,
                  generate_rsf_data_flag=True,
                  jlogz=c.jlogz,
                  jgx=c.jgx, sxbeg=c.sxbeg, gxbeg=c.gxbeg):
    
    if model_filename==None:
        model_filename=f"{folder}.hh"
    
    fig_path = f"../latex/Fig/test_{prefix}_{folder}"
    
    # expand model
    model_output="vel_test.rsf"
    print(model_output)
    vel_test = generate_model(model_input=model_filename, 
                              model_output=model_output, 
                              stretch_X=stretch_X,
                              random_state_number=c.random_state_number,
                              distort_flag=distort_flag,
                              crop_flag=False,
                              test_flag=True)
          
    # model data
    if generate_rsf_data_flag:
        cmd(f"mkdir {folder}")
        cmd(f"cp {model_output} {folder}/{model_output}")
        # check stability
        print(f"you chose dt = {dt}, dt < {dx/np.max(vel_test):.4f} should be chosen for stability \n")
        # force stability
        assert dt < dx/np.max(vel_test)
        generate_rsf_data(model_name=f"{folder}/vel_test.rsf", 
                          shots_out=f"{folder}/shots_cmp_test.rsf", 
                          logs_out=f"{folder}/logs_test.rsf")
    
    # read data
    X_test, T_test = read_rsf_XT(shots_rsf=f"{folder}/shots_cmp_test.rsf", 
                                              logs_rsf=f"{folder}/logs_test.rsf")
    
    nCMP = int(net_dict["0"].input.shape[3])
    # X_scaled, T_test = make_multi_CMP_inputs(X_scaled, T_test, nCMP_max)
    
    X_scaled, T_scaled = prepare_XT(X_test, T_test)
    T_test = T_scaler.inverse_transform(T_scaled)
    
    sample_reveal = nCMP_max+1
    plt_nb_T(1e3*np.concatenate((np.squeeze(X_scaled[sample_reveal,:,:,-1]), np.flipud(np.squeeze(X_scaled[sample_reveal,:,:,0]))), axis=0),
        title="CMP first | CMP last", dx=200, dz=1e3*dt*c.jdt,
        vmin=-0.1, vmax=0.1,
        origin_in_middle=True, ylabel="Time(s)", fname=f"{fig_path}_X_scaled", cbar_label = "")
    if nCMP == 1:
        X_scaled = X_scaled[:,:,:,nCMP_max//2:nCMP_max//2+1]
    
    # predict with all networks and save average
    T_pred_total = np.zeros_like(net_dict["0"].predict(X_scaled))    
    T_pred_dict = np.zeros((2*len(net_dict), T_pred_total.shape[0], T_pred_total.shape[1]))
    
    iNet=0
    for net in net_dict.values():
        T_pred_tmp = net.predict(X_scaled)
        T_pred_tmp = T_scaler.inverse_transform(T_pred_tmp)
        T_pred_dict[iNet,:,:] = T_pred_tmp
        T_pred_tmp = net.predict(np.flip(X_scaled, axis=3))
        T_pred_tmp = T_scaler.inverse_transform(T_pred_tmp)
        T_pred_dict[iNet+1,:,:] = T_pred_tmp
        iNet += 2
   
    T_pred = np.mean(T_pred_dict, axis=0)
     
    # interpolation for display
    ups_plot = c.ups_plot
    T_pred = upsample(T_pred, ups_plot)
    T_test = upsample(T_test, ups_plot)
    
    np_to_rsf(T_pred, f"{folder}/logs_pred.rsf", d1=25, d2=25)
    np_to_rsf(T_test, f"{folder}/logs_test_m.rsf", d1=25, d2=25)
    variance = np.var(T_pred_dict, axis=0)
    
    plt_nb_T(upsample(np.sqrt(variance), ups_plot), title="Standard deviation",
             dx=jgx*dx/ups_plot, dz=jlogz*dx/ups_plot,
             fname=f"{fig_path}_inverted_std_dev",
             vmin=0, vmax=1, figsize=(16,6))
    
    plt_nb_T(T_pred-T_test, title="Pred-True",
             dx=jgx*dx, dz=jlogz*dx,
             fname=f"{fig_path}_inverted_error",
             vmin=-1, vmax=1)
    
    plt_nb_T(T_pred, title=f"DL, R2 = {r2_score(T_test.flatten(), T_pred.flatten()):.2f}, NRMS={nrms(T_pred, T_test):.1f}%",
             dx=jgx*dx/ups_plot, dz=jgx*dx/ups_plot,
             vmin=np.min(1e-3*T_test), 
             vmax=np.max(1e-3*T_test),
             fname=f"{fig_path}_inverted", figsize=(16,6))
        
    plt_nb_T(T_test,
             dx=jgx*dx/ups_plot, dz=jgx*dx/ups_plot,
             vmin=np.min(1e-3*T_test), 
             vmax=np.max(1e-3*T_test),
             fname=f"{fig_path}_true",
             title=f"True model",
             figsize=(16,6))

#%%
def run_all_tests(net_dict=None, prefix="single", generate_rsf_data_flag=False):
    # Marmousi-based tests    
    test_on_model("marmvel1D", net_dict=net_dict, prefix=prefix, stretch_X=10, generate_rsf_data_flag=generate_rsf_data_flag)
    test_on_model("marmvel", net_dict=net_dict, prefix=prefix, stretch_X=1, generate_rsf_data_flag=generate_rsf_data_flag)
    test_on_model("marm2", net_dict=net_dict, prefix=prefix, stretch_X=1, generate_rsf_data_flag=generate_rsf_data_flag)
    # Overthrust-based tests
    test_on_model("overthrust1D", net_dict=net_dict, prefix=prefix, stretch_X=2, generate_rsf_data_flag=generate_rsf_data_flag)
    # cmd("sfadd < overthrust3D_orig.hh add=-1 | sfclip2 lower=1.5 --out=stdout > overthrust3D.hh")
    # cmd("sfwindow < overthrust3D_orig.hh n3=120 f1=400 n1=1 | sftransp | sfadd scale=1000 | sfput d1=25 d2=25 --out=stdout > overthrust_test_2D_1.hh")
    test_on_model("overthrust_test_2D_1", net_dict=net_dict, prefix=prefix, stretch_X=1, generate_rsf_data_flag=generate_rsf_data_flag)
    # cmd("sfwindow < overthrust3D.hh n3=120 f2=400 n2=1 | sftransp | sfadd scale=1000 | sfput d1=25 d2=25 --out=stdout > overthrust_test_2D_2.hh")
    test_on_model("overthrust_test_2D_2", net_dict=net_dict, prefix=prefix, stretch_X=1, generate_rsf_data_flag=generate_rsf_data_flag)
    # SEAM I based tests
    test_on_model("seam100", net_dict=net_dict, prefix=prefix, stretch_X=2, generate_rsf_data_flag=generate_rsf_data_flag)
    # cmd("sfwindow < SEAM_I_3D_20m.hh f3=100 n3=151 f1=1400 | sftransp memsize=100000 plane=13 | sfwindow f3=20 n3=1 f2=500 n2=1000 | sfput o1=0 o2=0 --out=stdout > seam_i_sediments.hh")
    test_on_model("seam_i_sediments", net_dict=net_dict, prefix=prefix, stretch_X=1, generate_rsf_data_flag=generate_rsf_data_flag)
    test_on_model("seam_karst", net_dict=net_dict, prefix=prefix, stretch_X=1, generate_rsf_data_flag=generate_rsf_data_flag)
    #cmd("sfwindow < SEAM_I_3D_20m.hh f3=10 n3=151 | sftransp memsize=100000 plane=23 | sftransp memsize=100000 plane=12 | sfwindow f3=1455 n3=1 --out=stdout > seam_i_salt.hh")
    test_on_model("seam_i_salt", net_dict=net_dict, prefix=prefix, stretch_X=2, generate_rsf_data_flag=generate_rsf_data_flag)
    test_on_model("seam_arid", net_dict=net_dict, prefix=prefix, stretch_X=1, generate_rsf_data_flag=generate_rsf_data_flag)
    
#run_all_tests(net_dict=singleCMP_net_dict, prefix="singleCMP", generate_rsf_data_flag=True)
run_all_tests(net_dict=multiCMP_net_dict, prefix="multiCMP", generate_rsf_data_flag=True)

print(f"Total execution time is {toc(tic_total)}")

#%% PLOT FWI RESULTS

# for folder in ["marm2",
#                "seam_i_sediments",
#                "seam100",
#                "overthrust"]:

for folder in ["overthrust"]:
    with cd(f"fwi_{folder}"):
        cmd("scons -j 4")
        fwi1 = rsf_to_np("fwi2.rsf")
        fwi2 = rsf_to_np("fwi_shi.rsf")
        velo = rsf_to_np("vel.rsf")
        velsm = rsf_to_np("smvel.rsf")
        R2o = r2_score(velo.flatten(), fwi2.flatten())
        fwi2 = resize(fwi2, (fwi2.shape[0], 120))
        fwi1 = resize(fwi1, (fwi2.shape[0], 120))
        plt_nb_T(fwi2, title=f"DL+MSFWI, R2={R2o:.2f}, NRMS={nrms(velo,fwi2):.1f}%", fname=f"../../latex/Fig/msfwi_{folder}", dx=25, dz=25, figsize=(32,6), vmin=1.5, vmax=4.5)
        plt_nb_T(velsm, 
                 title=f"DL, R2={r2_score(velo.flatten(),velsm.flatten()):.2f}, NRMS={nrms(velo,velsm):.1f}%", 
                 fname=f"../../latex/Fig/dl_{folder}", 
                 dx=25, dz=25, figsize=(16,6), vmin=1.5, vmax=4.5)
        plt_nb_T(velo, 
                 title=f"True model", 
                 fname=f"../../latex/Fig/true_{folder}", 
                 dx=25, dz=25, figsize=(16,6), vmin=1.5, vmax=4.5)

def plot_logs(log_x):    
    plt.figure(figsize=(11,18))
    depth = 0.025*np.array(range(120))
    plt.plot( 1e-3*velsm[log_x,:], depth, 'r', label="DL", linewidth=6)
    plt.plot(1e-3*fwi1[log_x,:], depth, 'b--', label="DL+FWI", linewidth=6)
    plt.plot(1e-3*fwi2[log_x,:], depth, 'bo', label="+MSFWI", markersize=15)
    plt.plot( 1e-3*velo[log_x,:], depth, 'black', label="True", linewidth=8, alpha=0.6)
    plt.ylabel("Depth (km)")
    plt.xlabel("Velocity (km/s)")
    plt.xlim((1.5, 4.5))
    plt.yticks([0,1,2,3])
    plt.title(f"Log at {int(0.025*log_x)} km")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.axis("tight")
    plt.savefig(f"../latex/Fig/log_{int(0.025*log_x)}")

plot_logs(240)
plot_logs(400)
plot_logs(480)


# %%
