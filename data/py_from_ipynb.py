#%% [markdown]
#  # Smart velocity analysis : mapping CMPs to velocity logs in laterally smooth media
#%%
#(c) Vladimir Kazei, Oleg Ovcharenko; KAUST 2019
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
import pydot
#plt.rcParams.update({'font.size': 5002})
#plt.rcParams['figure.figsize'] = [10, 5]
import seaborn
import tensorflow as tf
# images
from IPython import get_ipython
from keras import backend as K
from keras.utils import multi_gpu_model

from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.layers import (AveragePooling2D, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, MaxPool2D, Reshape)
from keras.models import load_model
from numpy.random import randint, seed
from scipy import ndimage
from skimage.transform import resize

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
#import styler
from myutils import cd, cmd, const, elastic_transform, plt_nb_T, toc, aug_flip, merge_dict, np_to_rsf, rsf_to_np, nrms

seed()
matplotlib.rc('image', cmap='RdBu_r')
seaborn.set_context('paper', font_scale=5)

CUDA_VISIBLE_DEVICES = "0,1,2,3"

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
random_model_repeat = 1000
stretch_X_train = 1
alpha_deform = 500
sigma_deform = 100
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
# 
# 1) generate seismic data set
# 
# 2) build neural network
# 
# 3) train neural network
# 
# 4) test it on a model that it has not seen
    

#%% [markdown]
# ## Model generation
# 
# we utilize common deep learning image augmentation technique -- elastic transform
# 
# code first:

#%%


#%%
# Fetch data from reproducibility.org
#sf.Fetch('marm','marmvel.hh')

# #%% [markdown]
#    ## Generating the model
#    First we create model by augmenting Marmousi II


def generate_model(model_input=const.trmodel, 
                   model_output="marm.rsf",
                   dx=const.dx,
                   stretch_X=1,
                   training_flag=False,
                   random_state_number=const.random_state_number,                  
                   distort_flag=True):
    # downscale marmousi
    #def rescale_to_dx(rsf_file_in, rsf_file_out, dx)
    model_orig = sf.Input(model_input)
    vel = model_orig.read()
    vel = np.concatenate((vel, np.flipud(vel), vel), axis = 0)
    if distort_flag:
        vel_log_res = vel
        #vel_log_res = resize(vel_log_res[:,:], (np.shape(vel)[0]//2, np.shape(vel)[1]//2))
        print(f"Random state number = {random_state_number}")
        #vel = resize(vel_log_res, vel.shape)
        np.random.RandomState(random_state_number)
        print(random_state_number)
        l0 = randint(np.shape(vel)[0])
       
        h0 = min(l0 + np.shape(vel)[0]//4 + randint(np.shape(vel)[0]//2), 
                 np.shape(vel)[0])
        l1 = randint(np.shape(vel)[1]//3)
        h1 = min(l1 + np.shape(vel)[1]//3 + randint(np.shape(vel)[1]//2), 
                 np.shape(vel)[1])
        print(l0, l1, h0, h1)
        vel_log_res = vel_log_res[l0:h0, l1:h1]
        
        vel = resize(vel_log_res, vel.shape)
    # we downscale
    scale_factor = dx / model_orig.float("d1")
    
    print(np.shape(vel))
    vel = resize(vel[:,:], (stretch_X*np.shape(vel)[0]//scale_factor, np.shape(vel)[1]//scale_factor))
    print(f"Model downscaled {scale_factor} times to {dx} meter sampling")
    if stretch_X != 1:
        print(f"Model stretched {stretch_X} times to {dx} meter sampling \n")
    
    # we concatenate horizontally, this is confusing because of flipped axis in madagascar
    
    vel = np.atleast_3d(vel)
    
    if distort_flag:
        vel = elastic_transform(vel, alpha_deform, sigma_deform, v_dx=dx, random_state_number=random_state_number)
    vel = np.squeeze(vel)
    vel_alpha = (0.9+0.2*random.random())
    print(vel_alpha)
    vel *= vel_alpha
    # add water
    # vel = np.concatenate((1500*np.ones((vel.shape[0], 20)), vel), 
    #                      axis=1)
    #vel = ndimage.median_filter(vel, size=(7,3))
    #vel = 1500 * np.ones_like(vel)
    print(f"Writing to {model_output}")
    np_to_rsf(vel, model_output)
    return vel

vel = generate_model(stretch_X=stretch_X_train, training_flag=False, random_state_number=randint(10000))
plt_nb_T(vel)

#%%
# Stretched Marmousi

vel = generate_model(stretch_X=stretch_X_train, distort_flag = False)
plt_nb_T(vel, fname="../latex/Fig/stretchMarm")
N = np.shape(vel)


#%%
# vel_example = elastic_transform(np.atleast_3d(vel), alpha_deform//2, sigma_deform, 
#                                 random_state_number=const.random_state_number, plot_name="Mild")
# vel_example = elastic_transform(np.atleast_3d(vel), alpha_deform, sigma_deform, 
#                                 random_state_number=const.random_state_number, plot_name="Normal")


#vel_example = elastic_transform(np.atleast_3d(vel), 
#                                alpha_deform, sigma_deform, 
#                                random_state_number=random_state_number, 
#                                plot_name="Normal")
vel_example = generate_model(stretch_X=stretch_X_train, training_flag=True, random_state_number=const.random_state_number)
vel1 = generate_model(stretch_X=stretch_X_train, training_flag=False, random_state_number=randint(10000))
vel2 = generate_model(stretch_X=stretch_X_train, training_flag=False, random_state_number=randint(10000))
vel3 = generate_model(stretch_X=stretch_X_train, training_flag=False, random_state_number=randint(10000))
vel4 = generate_model(stretch_X=stretch_X_train, training_flag=False, random_state_number=randint(10000))
plt_nb_T(np.concatenate((vel_example, vel1, vel2, vel3, vel4), axis=1), fname="../latex/Fig/random_model_example")

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
# ## Generator features
# 
# ### Several layers can be generated from a single one
# 
# ### Velocities are exactly the same as in mother-model (Marmousi)
#%% [markdown]
# ## Modeling data with constant offset on GPU with Madagascar

#%%
# Setting up parameters
N = np.shape(vel)
dt = const.dt
dx = const.dx
T_max = const.T_max
nt = int(T_max / dt + 1)

print(f"number of time steps = {nt}")



# check stability
print(f"you chose dt = {dt}, dt < {dx/np.max(vel):.4f} should be chosen for stability \n")
# force stability
assert dt < dx/np.max(vel)

# ricker wavelet is roughly bounded by 3f_dominant
# therefore the sampling rate principally acceptable sampling rate would be

central_freq = const.central_freq
print(f"dt from Nyquist criterion is {1/(2*3*central_freq)}")

jgx = const.jgx
jsx = const.jsx
jdt = const.jdt
jlogz = const.jlogz

print(f"dt chosen for CNN is {jdt*dt}, which is {(1/(3*central_freq))/(jdt*dt)} samples per cycle")
sxbeg = const.sxbeg 
gxbeg = const.gxbeg 
szbeg = const.szbeg  


#%%


# model data and sort into CMPs function
def generate_rsf_data(model_name="marm.rsf", central_freq=central_freq, dt=dt, dx=const.dx, 
                        nt=nt, sxbeg=sxbeg, gxbeg=gxbeg, szbeg=szbeg, 
                        jsx=jsx, jgx=jgx, jdt=jdt,
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
                            f"sxbeg={sxbeg} chk=n szbeg=2 jsx={jgx} jsz=0 gxbeg={gxbeg} gzbeg={szbeg} jgx={jgx} jgz=0 > shots.rsf"))
    print(f"Modeling time for {ns} shots = {time.time()-t_start}")
    if full_shots_out != None:
        cmd(f"sfcp < shots.rsf > {full_shots_out}")

    #   ## Analyze and filter the data set generated
    # correct header and reduce sampling in time jdt (usually 4) times
    cmd(f"sfput < shots.rsf d3={jgx*dx} | sfwindow j1={jdt} | sfbandpass flo=2 fhi=4 > shots_decimated.rsf")
    cmd(f"sfrm shots.rsf")
    # sort into cmp gathers and discard odd cmps and not full cmps
    cmd(f"sfshot2cmp < shots_decimated.rsf half=n | sfwindow j3=2 min3={(-1.5*gxbeg+1.5*sxbeg)*dx} max3={(Nx-(2.5*sxbeg-.5*gxbeg))*dx} > {shots_out}")
    cmd(f"sfrm shots_decimated.rsf")
    # create the logs -- training outputs
    cmd(f"sfwindow < {model_name} min2={(-gxbeg+2*sxbeg)*dx} j2={jsx} max2={(Nx-(2*sxbeg-gxbeg))*dx} > {logs_out}")
    cmd(f"sfin < {logs_out}")
    return 0

#@profile
def generate_rsf_data_multi(model_name="marm.rsf", central_freq=central_freq, dt=dt, 
                        nt=nt, sxbeg=sxbeg, gxbeg=gxbeg, szbeg=szbeg, 
                        jsx=jsx, jgx=jgx, jdt=jdt,
                        logs_out="logs.rsf", shots_out="shots_cmp.rsf", iShotBlock=None): 
    cmd(f"mkdir /dev/shm/RSFTMP/data_{iShotBlock}")
    cmd(f"chmod 777 /dev/shm/RSFTMP/data_{iShotBlock}")    
    os.environ["DATAPATH"]=f"/dev/shm/RSFTMP/data_{iShotBlock}"
    cmd(f"echo $DATAPATH")
    cmd(f"mkdir data_{iShotBlock}")
    seed()
    #cmd(f"sfwindow < overthrust3D.hh n3=120 f1={iShotBlock*randint(0,1e7) % 400} n1=1 | sftransp | sfadd scale=1000 | sfput d1=25 d2=25 --out=stdout > data_{iShotBlock}/overthrust2D.hh")
    #cmd(f"cp {const.trmodel} data_{iShotBlock}/")
    with cd(f"data_{iShotBlock}"):
        _vel = generate_model(model_input=f"../{const.trmodel}", random_state_number=(iShotBlock + randint(0,1e7)))
        #plt_nb_T(_vel)
        generate_rsf_data()

#%%
#@profile
def generate_all_data(random_model_repeat=random_model_repeat):

    K.clear_session()
    start_modeling_time = time.time()
    procs = []
    for iShotBlock in range(random_model_repeat):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(iShotBlock % 4)
        proc = multiprocessing.Process(target=generate_rsf_data_multi, kwargs ={'iShotBlock' : iShotBlock})
        proc.start()
        procs.append(proc)
        if len(procs) > 100:
            for proc in procs:
                proc.join()
            procs = []
    for proc in procs:
        proc.join()
    print(f"Time for modeling = {toc(start_modeling_time)}")
    
    start_merging_rsf_time = time.time()        
    cmd(f"sfcat data_*/shots_cmp.rsf axis=3 > shots_cmp_full.rsf")
    cmd(f"sfcat data_*/logs.rsf axis=2 > logs_full.rsf")
    
    print(f"Time for merging rsf files = {toc(start_merging_rsf_time)}")
    os.environ["CUDA_VISIBLE_DEVICES"]=CUDA_VISIBLE_DEVICES
    
if generate_rsf_data_flag:
    generate_all_data()
#%% [markdown]
# ## Read data into numpy and check that the number of logs is the same as number of shots

#%%
def read_rsf_to_np(shots_rsf='shots_cmp_full.rsf', logs_rsf='logs_full.rsf',
                  n_offsets=None, j_log_z=jlogz):
    shots_cmp = sf.Input(shots_rsf)
    X_data = shots_cmp.read()
    
    if n_offsets==None :
        X_data = X_data[:,:(np.shape(X_data)[1] + 1) // 2,:]
    
    X_data = np.expand_dims(X_data, axis=3)
    X_size = np.shape(X_data)
    logs = sf.Input(logs_rsf)
    T_data = logs.read()
    
    # decimate logs in vertical direction --2 times by default
    T_data = resize(T_data, (np.shape(T_data)[0], np.shape(T_data)[1] // j_log_z))
    T_size = np.shape(T_data)
    print(T_size)

    # ensure that the number of logs is equal to the number of CMPs
    assert (X_size[0] == T_size[0])
    return X_data, T_data

X_data, T_data = read_rsf_to_np(shots_rsf='shots_cmp_full.rsf', logs_rsf='logs_full.rsf')

plt_nb_T(X_data[:,-3,:,0], title="Common offset (600 m) gather", dx=const.dx*jgx*2, dz=1e3*dt*jdt, 
        origin_in_middle=True, ylabel="Time(s)", fname="../latex/Fig/X_data_short_offset")
#X_data=X_data[:,:-3,:,:]
#plt_nb_T(T_data, dx=jsx*dx)

#%% [markdown]
#    # DATA is prepared, ML starts here
#    ## Standard rescaling first 

T_scaler = MinMaxScaler([-1,1])
T_scaler.fit(T_data)
T_scaled = T_scaler.transform(T_data)
#T_scaled = T_data/1e3

def scale_X_data(X_data_test=X_data, X_scaler=None):
    generate_scaler = False
    if X_scaler == None:
        generate_scaler = True
        # custom scaling of X_data
        X_scaler = StandardScaler() # MinMaxScaler([-1, 1])
        print(np.shape(X_data))
        X_data_cut=X_data[:,:-3,:,:]
        #plt_nb_T(1e3*np.squeeze(np.percentile(abs(X_data_cut), axis=0, q=0.01)), vmax=1e-8, title="Percentile")
        X_matrix = X_data_cut.reshape([X_data_cut.shape[0], -1])
        X_scaler.fit(X_matrix)
        #X_scaler.scale_=np.ones_like(X_scaler.scale_)
        plt_nb_T(np.flipud(1e3*X_scaler.scale_.reshape([X_data_cut.shape[1], -1])), title="Scale",ylabel="Time(s)", dx=200, dz=1e3*dt*jdt, cbar_label="", fname="../latex/Fig/X_scale")

        plt_nb_T(np.flipud(1e3*X_scaler.mean_.reshape([X_data_cut.shape[1], -1])), title="Mean", ylabel="Time(s)", dx=200, dz=1e3*dt*jdt, cbar_label="", fname="../latex/Fig/X_mean")
        plt_nb_T(np.flipud(np.log(X_scaler.var_.reshape([X_data_cut.shape[1], -1]))), title="Variance", ylabel="Time(s)", dx=200, dz=1e3*dt*jdt, cbar_label="", fname="../latex/Fig/X_log_var")

    X_data_test = X_data_test[:,:-3,:,:]
    X_matrix_test = X_data_test.reshape([X_data_test.shape[0], -1])
    X_data_test_matrix_scaled = X_scaler.transform(X_matrix_test)
    X_data_test_scaled = 0.1 * X_data_test_matrix_scaled.reshape(X_data_test.shape)
    X_data_test_scaled = np.clip(X_data_test_scaled, a_min=-1, a_max=1)
    plt_nb_T(1e3*np.squeeze(np.percentile(abs(X_data_test), axis=0, q=10)), title="Percentile")

    if generate_scaler:
        return X_data_test_scaled, X_scaler
    return X_data_test_scaled

X_scaled, X_scaler = scale_X_data(X_data)


#%%
# expand to multiple CMPs as channels in input
#@profile
def make_multi_CMP_inputs(X_data, T_data, nCMP, n_models=1):
    # first we prepare array for X_data_multi_CMP
    X_data_multi_CMP = np.zeros((n_models, 
                                 X_data.shape[0]//n_models-nCMP+1,
                                 X_data.shape[1],
                                 X_data.shape[2],
                                 nCMP)).astype("float32")
    
    # add model dimension
    X_data = X_data.reshape(n_models, X_data.shape[0]//n_models, X_data.shape[1], X_data.shape[2], 1).astype("float32")
    T_data = T_data.reshape(n_models, T_data.shape[0]//n_models, T_data.shape[1])
    
    for i in range(nCMP-1):
        print(f"Expanding to multiCMP inputs channel {i} out of {nCMP}")
        X_data_multi_CMP[:,:,:,:,i] = X_data[:,i:-nCMP+i+1,:,:,0]    
    X_data_multi_CMP[:,:,:,:,nCMP-1] = X_data[:,nCMP-1:,:,:,0]
    
    if nCMP==1 :
        T_data_multi_CMP = T_data
    else :
        T_data_multi_CMP = T_data[:, (nCMP-1)//2 : -(nCMP-1)//2, :]
        
    X_data_multi_CMP = X_data_multi_CMP.reshape(X_data_multi_CMP.shape[0]*X_data_multi_CMP.shape[1], 
                                                X_data_multi_CMP.shape[2], 
                                                X_data_multi_CMP.shape[3],
                                                X_data_multi_CMP.shape[4])
    T_data_multi_CMP = T_data_multi_CMP.reshape(T_data_multi_CMP.shape[0]*T_data_multi_CMP.shape[1], 
                                                T_data_multi_CMP.shape[2])
    assert (X_data_multi_CMP.shape[0] == T_data_multi_CMP.shape[0])
    return X_data_multi_CMP, T_data_multi_CMP

nCMP = 21
X_scaled_multi, T_scaled_multi = make_multi_CMP_inputs(X_scaled, T_scaled, nCMP, n_models=random_model_repeat)

plt_nb_T(T_data, dx=jgx*dx, dz=jlogz*dx, fname="../latex/Fig/T_data")
plt_nb_T(1e3*T_scaled, dx=jgx*dx, dz=jlogz*dx, fname="../latex/Fig/T_scaled")

# extract central CMPs for singleCMP network
X_scaled = X_scaled_multi[:,:,:,nCMP//2:nCMP//2+1]
T_scaled = T_scaled_multi

#%%
# show single training sample
sample_reveal = nCMP
plt_nb_T(1e3*np.concatenate((np.squeeze(X_scaled_multi[sample_reveal,:,:,-1]), np.flipud(np.squeeze(X_scaled_multi[sample_reveal,:,:,0]))), axis=0),
        title="CMP first | CMP last", dx=200, dz=1e3*dt*jdt, 
        origin_in_middle=True, ylabel="Time(s)", fname="../latex/Fig/X_scaled", cbar_label = "")
plt_nb_T(1e3*np.concatenate((np.squeeze(X_data[sample_reveal,:,:]), np.flipud(np.squeeze(X_data[sample_reveal+nCMP,:,:]))), axis=0),
        title="CMP first | CMP last", dx=200, dz=1e3*dt*jdt, 
        origin_in_middle=True, ylabel="Time(s)", fname="../latex/Fig/X_raw", cbar_label = "")
print(np.shape(1e3*T_scaled[sample_reveal-(nCMP+1)//2:sample_reveal+(nCMP-1)//2:nCMP]))

plt_nb_T(1e3*T_data[sample_reveal-(nCMP-1)//2:sample_reveal+(nCMP-1)//2,:], 
         dx=100, dz=const.dx*jlogz, 
         title="scaled velocity logs")

#%% [markdown]
#    # CNN construction single CMP -> log under the CMP

# 1D total variation for the output
def tv_loss(y_true, y_pred):
    #b, h, w, c = img.shape.as_list()
    a = K.abs(y_pred[:, :-1] - y_pred[:, 1:])
    tv = 0*K.mean(a, axis=-1)
    total = tv + K.mean(K.square(y_pred - y_true), axis=-1)
    return total

def create_model(inp_shape, out_shape, jlogz=jlogz):
    model = keras.models.Sequential()
    activation = 'elu'
    activation_dense = activation
    padding = 'same'
    kernel_size = (3, 7)
    model.add(Conv2D(filters=32, kernel_size=kernel_size, activation=activation, padding=padding, input_shape=inp_shape))
    model.add(BatchNormalization())      
    model.add(Conv2D(filters=32, kernel_size=kernel_size, strides=(2,2), activation=activation, padding=padding))       
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=kernel_size, strides=(2,2), activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=kernel_size, strides=(2,2), activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=16, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=16, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=8, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=8, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=4, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=4, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=2, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=2, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=1, kernel_size=(3, 15), activation='linear', padding="valid"))
    model.add(Flatten())
    # model.add(Dense(2*out_shape[0], activation=activation_dense))
    # model.add(Dropout(dropout))
    # model.add(Dense(out_shape[0], activation=activation_dense))
    # model.add(Dense(out_shape[0], activation='linear'))
    # model = multi_gpu_model(model, gpus=2)
    model.compile(loss=tv_loss,
                  optimizer=keras.optimizers.Nadam(),
                  metrics=['accuracy'])
    return model

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
def batch_generator(X, Y, batch_size = None):
    indices = np.arange(len(X)) 
    batch=[]
    while True:
            # it might be a good idea to shuffle your data before each epoch
            np.random.shuffle(indices) 
            for i in indices:
                batch.append(i)
                if len(batch)==batch_size//2:
                    yield np.concatenate((np.flip(X[batch], axis=3), X[batch]), axis=0), np.concatenate((Y[batch], Y[batch]), axis=0)
                    batch=[]

#%%
# Init callbacks
#@profile
def train_model(prefix="single", X_scaled=None, T_scaled=None, weights=None):
    X_scaled = X_scaled.astype("float32")
    T_scaled = T_scaled.astype("float32")
    net = create_model(np.shape(X_scaled)[1:], np.shape(T_scaled)[1:])
    net.summary()
    if weights != None:
        net.load_weights(weights)
    
    
    # keras.utils.plot_model(net, to_file=f"../latex/Fig/net_{prefix}.png",
    #                        show_shapes=True, show_layer_names=True)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=7)
    model_checkpoint = ModelCheckpoint("trained_net",
                                       monitor='val_loss',
                                       save_best_only=True,
                                       period=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=3, min_lr=1e-5, verbose=1)
    

    # Split the data
    X_t, X_test, T_t, T_test = train_test_split(X_scaled, T_scaled, test_size=0.1, shuffle=False)
    X_test_train, X_test_test, T_test_train, T_test_test = train_test_split(X_test, T_test, test_size=0.5, shuffle=False)
    X_train, X_valid, T_train, T_valid = train_test_split(X_t, T_t, test_size=0.2, shuffle=False)
    
    X_train = np.concatenate((X_train, X_test_train), axis=0)
    T_train = np.concatenate((T_train, T_test_train), axis=0)
    print(np.shape(X_valid))
    
    # TRAINING
    batch_size = 64
    steps_per_epoch = len(X_train)//batch_size
    print(f"Batches per epoch = {steps_per_epoch}")
    history = net.fit_generator(batch_generator(X_train, T_train, batch_size=batch_size),
                      validation_data=(X_valid, T_valid),
                      epochs=100,
                      verbose=2,
                      shuffle=True,
                      max_queue_size=200,
                      workers=10, 
                      use_multiprocessing=False,
                      steps_per_epoch = steps_per_epoch,
                      callbacks=[
                          model_checkpoint,
                          reduce_lr,
                          early_stopping])    
    
    print("Optimization Finished!")
        
    # fitting the training set
    plt_nb_T(T_scaler.inverse_transform(T_test), dx=100, dz=jlogz*dx,
             title="Training |  Testing",
             fname=f"../latex/Fig/train_{prefix}_true",
             split_line=True,
             vmin=1.5, vmax=5.5)
    # plt_nb_T(T_test, dx=100, dz=jlogz*dx,
    #          title="Training |  Testing",
    #          fname=f"../latex/Fig/train_{prefix}_true",
    #          split_line=True,
    #          vmin=1.5, vmax=5.5)
    
    print(net.evaluate(X_test_test, T_test_test))
    
    plt_nb_T(T_scaler.inverse_transform(net.predict(X_test)), dx=100, dz=jlogz*dx,
                 fname=f"../latex/Fig/train_{prefix}_predicted",
                 title=f"NRMS={nrms(T_scaler.inverse_transform(net.predict(X_test_train)), T_scaler.inverse_transform(T_test_train)):.1f}  "+
                 f" |  NRMS={nrms(T_scaler.inverse_transform(net.predict(X_test_test)), T_scaler.inverse_transform(T_test_test)):.1f}",
                 split_line=True,
                 vmin=1.5, vmax=5.5)
    # plt_nb_T(net.predict(X_test), dx=100, dz=jlogz*dx,
    #              fname=f"../latex/Fig/train_{prefix}_predicted",
    #              title=f"NRMS={nrms(net.predict(X_test_train), T_test_train):.1f}  "+
    #              f" |  NRMS={nrms(net.predict(X_test_test), T_test_test):.1f}",
    #              split_line=True,
    #              vmin=1.5, vmax=5.5)
    
    return net, history

def load_history(fname_history):
    with open(fname_history,'rb') as f:
        return pickle.load(f)

def save_history(history, fname_history):
    with open(fname_history,'wb') as f:
        pickle.dump(history, f)

def train_ensemble(prefix, X_scaled, T_scaled):
    valid_best=1
    net_dict = {}
    for iNet in range(1):
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
        
        if retrain_flag:
            history_best = merge_dict(history_prev, history_best)
        
        net_best.save_weights(f"{prefix}_weights.h5")
        save_history(history_best, f"history_{prefix}")
        
        plt.figure(figsize=(16,9))
        plt.semilogy(history_best['loss'],'b--', label='Training loss')
        plt.semilogy(history_best['val_loss'],'r', label='Validation loss')
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(f"../latex/Fig/{prefix}_loss", bbox_inches='tight')
        plt.grid(True,which="both",ls="-")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        
    return net_dict, net_best, history_best

singleCMP_net_dict, singleCMP_net_best, history_best = train_ensemble("singleCMP", X_scaled, T_scaled)
multiCMP_net_dict, multiCMP_net_best, history_best = train_ensemble("multiCMP", X_scaled_multi, T_scaled_multi)
       

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
#%%
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
                  net_dict=singleCMP_net_dict,
                  prefix="singleCMP",
                  model_filename=None, 
                  distort_flag=False,
                  stretch_X=None,
                  nCMP_max=nCMP,
                  generate_rsf_data_flag=True,
                  jgx=jgx, sxbeg=sxbeg, gxbeg=gxbeg,
                  X_scaler=X_scaler):
    
    if model_filename==None:
        model_filename=f"{folder}.hh"
    
    fig_path = f"../latex/Fig/test_{prefix}_{folder}"
    
    # expand model
    model_output="vel_test.rsf"
    print(model_output)
    vel_test = generate_model(model_input=model_filename, 
                              model_output=model_output, 
                              stretch_X=stretch_X,
                              random_state_number=const.random_state_number,
                              distort_flag=distort_flag)
          
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
    X_data_test, T_data_test = read_rsf_to_np(shots_rsf=f"{folder}/shots_cmp_test.rsf", 
                                              logs_rsf=f"{folder}/logs_test.rsf")
    
    # X_scaled
    X_scaled = scale_X_data(X_data_test, X_scaler)
    
    nCMP = int(net_dict["0"].input.shape[3])
    X_scaled, T_data_test = make_multi_CMP_inputs(X_scaled, T_data_test, nCMP_max)
    sample_reveal = nCMP_max+1
    plt_nb_T(1e3*np.concatenate((np.squeeze(X_scaled[sample_reveal,:,:,-1]), np.flipud(np.squeeze(X_scaled[sample_reveal,:,:,0]))), axis=0),
        title="CMP first | CMP last", dx=200, dz=1e3*dt*jdt, 
        origin_in_middle=True, ylabel="Time(s)", fname=f"{fig_path}_X_scaled", cbar_label = "")
    if nCMP == 1:
        X_scaled = X_scaled[:,:,:,nCMP_max//2:nCMP_max//2+1]
    
    # predict with all networks and save average
    T_pred_total = np.zeros_like(net_dict["0"].predict(X_scaled))    
    T_pred_dict = np.zeros((len(net_dict), T_pred_total.shape[0], T_pred_total.shape[1]))
    
    iNet=0
    for net in net_dict.values():
        T_pred_tmp = net.predict(X_scaled)
        T_pred_tmp = T_scaler.inverse_transform(T_pred_tmp)
        T_pred_dict[iNet,:,:] = T_pred_tmp
        iNet += 1
   
    T_pred = np.mean(T_pred_dict, axis=0)
    variance = np.var(T_pred_dict, axis=0)
    
    
    plt_nb_T(np.sqrt(variance), title="Standard deviation",
             dx=jgx*dx, dz=jlogz*dx,
             fname=f"{fig_path}_inverted_std_dev",
             vmin=0.05, vmax=1)
    
    # plt_nb_T(T_pred-T_data_test, title="Pred-True",
    #          dx=jgx*dx, dz=jlogz*dx,
    #          fname=f"{fig_path}_inverted_std_dev",
    #          vmin=-1, vmax=1)
    
    plt_nb_T(T_pred, title=f"{prefix} estimate, NRMS={nrms(T_pred, T_data_test):.1f}%",
             dx=jgx*dx, dz=jlogz*dx,
             vmin=np.min(1e-3*T_data_test), 
             vmax=np.max(1e-3*T_data_test),
             fname=f"{fig_path}_inverted")
        
    plt_nb_T(T_data_test, 
             dx=jgx*dx, dz=jlogz*dx,
             fname=f"{fig_path}_true",
             title="True")
    
    print(np.shape(1e3*T_scaled[sample_reveal-(nCMP+1)//2:sample_reveal+(nCMP-1)//2:nCMP]))

#%
def run_all_tests(net_dict=singleCMP_net_dict, prefix="single", generate_rsf_data_flag=False):
    # Marmousi-based tests    
    test_on_model("marmvel1D", net_dict=net_dict, prefix=prefix, stretch_X=10, generate_rsf_data_flag=generate_rsf_data_flag)
    cmd("cp marmvel1D.hh marmvel1D_distort.hh")
    test_on_model("marmvel1D_distort", net_dict=net_dict, prefix=prefix, stretch_X=10, distort_flag=True, generate_rsf_data_flag=generate_rsf_data_flag)
    test_on_model("marmvel", net_dict=net_dict, prefix=prefix, stretch_X=1, generate_rsf_data_flag=generate_rsf_data_flag)
    test_on_model("marm2", net_dict=net_dict, prefix=prefix, stretch_X=1, generate_rsf_data_flag=generate_rsf_data_flag)
    # Overthrust-based tests
    test_on_model("overthrust1D", net_dict=net_dict, prefix=prefix, stretch_X=2, generate_rsf_data_flag=generate_rsf_data_flag)
    cmd("sfadd < overthrust3D_orig.hh add=-1 | sfclip2 lower=1.5 --out=stdout > overthrust3D.hh")
    cmd("sfwindow < overthrust3D.hh n3=120 f1=400 n1=1 | sftransp | sfadd scale=1000 | sfput d1=25 d2=25 --out=stdout > overthrust_test_2D_1.hh")
    test_on_model("overthrust_test_2D_1", net_dict=net_dict, prefix=prefix, stretch_X=1, generate_rsf_data_flag=generate_rsf_data_flag)
    cmd("sfwindow < overthrust3D.hh n3=120 f2=400 n2=1 | sftransp | sfadd scale=1000 | sfput d1=25 d2=25 --out=stdout > overthrust_test_2D_2.hh")
    test_on_model("overthrust_test_2D_2", net_dict=net_dict, prefix=prefix, stretch_X=1, generate_rsf_data_flag=generate_rsf_data_flag)
    # Slices of overthrust
    test_on_model("seam100", net_dict=net_dict, prefix=prefix, stretch_X=1, generate_rsf_data_flag=generate_rsf_data_flag)
    
run_all_tests(net_dict=singleCMP_net_dict, prefix="singleCMP", generate_rsf_data_flag=True)
run_all_tests(net_dict=multiCMP_net_dict, prefix="multiCMP")


#%%
print(f"Total execution time is {toc(tic_total)}")

# %%
