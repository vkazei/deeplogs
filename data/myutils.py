#%%
# utilities
import subprocess
import os
import matplotlib
import matplotlib.pyplot as plt
import time
import numpy as np
from numpy import linalg
import m8r as sf
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from tensorflow.python.ops.image_ops_impl import _random_flip
from skimage.transform import resize

class _const():
    """Default settings for modeling and inversion
    """
    dx = 50
    dt = 0.005
    T_max = 7
    nt = int(T_max / dt + 1)
    central_freq = 7
    jgx = 2
    jsx = jgx
    jdt = 4
    sxbeg = 5000//dx
    gxbeg = 1000//dx
    szbeg = 2
    jlogz = 2
    trmodel = "marmvel.hh"
    random_state_number = 314
    random_model_repeat = 100
    # upsample for plotting
    ups_plot = 4
    # one can stretch training models horizontally 
    stretch_X_train = 1

const = _const()

#%%
def tf_random_flip_channels(image, seed=None):
  """
  With a 1 in 2 chance, outputs the contents of `image` flipped along the
  third dimension, which is `channels`.  Otherwise output the image as-is.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or
           3-D Tensor of shape `[height, width, channels]`.
    seed: A Python integer. Used to create a random seed. See
      `tf.set_random_seed`
      for behavior.

  Returns:
    A tensor of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  """
  return _random_flip(image, 2, seed, 'random_flip_channels')

def upsample(X, upscale):
    return resize(X, upscale * np.array(X.shape))

def nrms(T_pred, T_true):
    return 100*linalg.norm(T_pred-T_true)/linalg.norm(T_true)

def rsf_to_np(file_name):
    f = sf.Input(file_name)
    vel = f.read()
    return vel

def np_to_rsf(vel, model_output, d1 = const.dx, d2 = const.dx):
    ''' Write 2D numpy array vel to rsf file model_output '''
    yy = sf.Output(model_output)
    yy.put('n1',np.shape(vel)[1])
    yy.put('n2',np.shape(vel)[0])
    yy.put('d1',d1)
    yy.put('d2',d2)
    yy.put('o1',0)
    yy.put('o2',0)
    yy.write(vel)
    yy.close()
    
def merge_dict(dict1, dict2):
    ''' Merge dictionaries with same keys'''
    dict3 = dict1.copy()
    for key, value in dict1.items():
        dict3[key] = np.concatenate((value, dict2[key]), axis=0)
    return dict3

def cmd(command):
    """Run command and pipe what you would see in terminal into the output cell
    """
    process = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    while True:
        output = process.stderr.readline().decode('utf-8')
        if output == '' and process.poll() is not None:
            # this prints the stdout in the end
            output2 = process.stdout.read().decode('utf-8')
            print(output2.strip())
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def plt_show_proceed(delay=1):
    plt.show(block=False)
    plt.pause(delay)
    plt.close()

def plt_nb_T(vel, fname="Velocity", title="",
             ylabel="Depth (km)", xlabel="Distance (km)",
             cbar=True,
             cbar_label = "(km/s)",
             vmin=None, vmax=None,
             split_line=False,
             dx=const.dx, dz=const.dx, no_labels=False, origin_in_middle=False,
             figsize=(16,9),
             xticks=True):
    plt.figure(figsize=figsize)
    vel_image = vel[:,:].T
    extent=(0, dx * vel.shape[0] * 1e-3, dz * vel.shape[1] *1e-3, 0)
    if origin_in_middle:
        extent = (-dx * vel.shape[0] * .5e-3, dx * vel.shape[0] * .5e-3, dz * vel.shape[1] *1e-3, 0)
    plt.imshow(vel_image * 1e-3, origin='upper', extent=extent)
    #plt.axis("equal")
    plt.axis("tight")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not xticks:
        plt.xticks([])
    plt.title(title)
    plt.clim(vmin,vmax)
    if cbar==True:
        cbar = plt.colorbar()
        cbar.ax.set_ylabel(cbar_label)
    if split_line:
        plt.axvline(x=extent[1]/2, color='black', linewidth=10, linestyle='-')
    
    if no_labels:
        plt.xlabel("")
        plt.axis('off')
        
    plt.savefig(fname, bbox_inches='tight')
    plt_show_proceed()

def toc(start_time):
    return (time.time() - start_time)

def aug_flip(vel):
    vel = np.concatenate((vel, np.flipud(vel),vel), axis = 0)
    return vel

# to distort the model
def elastic_transform(image, alpha, sigma, random_state_number=None, v_dx=const.dx, plot_name=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
  
    random_state = np.random.RandomState(random_state_number)

    shape = image.shape
    #print(shape)
    
    # with our velocities dx is vertical shift
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), (sigma, sigma/10, 1), mode="constant", cval=0) * 4 * alpha
    
    # with our velocities dy is horizontal
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), (sigma, sigma/10, 1),  mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)
    
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)),  np.reshape(z, (-1, 1))
    
    distorted_image = map_coordinates(image, indices, order=1, mode='reflect', prefilter=False)
    distorted_image = distorted_image.reshape(image.shape)
    
    if plot_name != None:
        plt_nb_T(v_dx * np.squeeze(dx[:,:]), fname=f"VerticalShifts_{alpha}", title="Vertical shifts (km)")
        dq_x = 100
        dq_z = 17
        M = np.hypot(dy.squeeze()[::dq_x,::dq_z].T, dx.squeeze()[::dq_x,::dq_z].T)
        M = dx.squeeze()[::dq_x,::dq_z].T
        M = np.squeeze(image)[::dq_x,::dq_z].T
        if 1:
            fig1, ax1 = plt.subplots(figsize=(16,9))
            ax1.set_title('Guiding model')
            plt.imshow(1e-3*np.squeeze(image.T), extent=(0, v_dx * dx.shape[0] * 1e-3, v_dx * dx.shape[1] *1e-3, 0))
            plt.axis("tight")
            plt.xlabel("Distance (km)")
            plt.ylabel("Depth (km)")
            plt.colorbar()
            Q = ax1.quiver(
            1e-3*v_dx *y.squeeze()[::dq_x,::dq_z].T, 1e-3*v_dx *x.squeeze()[::dq_x,::dq_z].T, 
            np.abs(1e-4*v_dx*dx.squeeze()[::dq_x,::dq_z].T), 1e-3*v_dx*dx.squeeze()[::dq_x,::dq_z].T, 
            scale_units='xy', scale=1, pivot='tip')
            plt.savefig(f"../latex/Fig/shiftsVectors", bbox_inches='tight')
            plt_show_proceed()

        fig1, ax1 = plt.subplots(figsize=(16,9))
        ax1.set_title('Distorted model')
        plt.imshow(1e-3*np.squeeze(distorted_image.T), extent=(0, v_dx * dx.shape[0] * 1e-3, v_dx * dx.shape[1] *1e-3, 0))
        plt.axis("tight")
        plt.xlabel("Distance (km)")
        plt.ylabel("Depth (km)")
        plt.colorbar()
        Q = ax1.quiver(
            1e-3*v_dx *y.squeeze()[::dq_x,::dq_z].T, 1e-3*v_dx *x.squeeze()[::dq_x,::dq_z].T, 
            np.abs(1e-4*v_dx*dx.squeeze()[::dq_x,::dq_z].T), 1e-3*v_dx*dx.squeeze()[::dq_x,::dq_z].T, 
            scale_units='xy', scale=1, pivot='tip')
        plt.savefig(f"../latex/Fig/deformedModel{plot_name}", bbox_inches='tight')
        plt_show_proceed()

    return distorted_image