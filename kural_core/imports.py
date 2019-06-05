# libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
import pickle
import time
from nd2reader import ND2Reader
import warnings
from tqdm import tqdm

# functions
from scipy.signal import convolve2d
from skimage.measure import block_reduce, find_contours
from matplotlib.patches import Rectangle
from scipy.misc import imresize
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.spatial import ConvexHull
from skimage import measure
from matplotlib import animation, rc
from IPython.display import HTML, display, clear_output
from os import listdir
from ipywidgets import widgets, Layout
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.gridspec as gridspec
from skimage.restoration import denoise_nl_means, estimate_sigma
from numpy.random import choice

# my stuff
from kural_core.models import *
from kural_core.data_processing import *
from kural_core.functions import *
from kural_core.train_model import train_model
