from torch.utils.data import Dataset
import os
import numpy as np
from scipy.misc import imresize
import torch
from PIL import Image
import torchvision
SCALE_FACTOR = 2**16-1

class PooledImageDataset(Dataset):
    def __init__(self, img, num_frames, num_angles, rnn_length=4):
        self.img = img
        self.nf = num_frames
        self.rnn_length = rnn_length
        self.frame_mod = rnn_length-1
        self.na = num_angles
        
    def __len__(self):
        return (self.nf-self.frame_mod)*self.na#*8
    
    def __getitem__(self,idx):
        transform_number = idx//(self.nf-self.frame_mod)
        addon = self.frame_mod*transform_number
        return self.img[idx+addon:idx+addon+self.rnn_length]

class AllImageRNNDataset(Dataset):
    def __init__(self, img, num_frames_list, num_angles, rnn_length=4):
        self.img = img
        self.nf = num_frames_list # Should be 1D numpy array of cumulative sum of frames
        self.rnn_length = rnn_length
        self.frame_mod = rnn_length-1
        self.na = num_angles
        self.cpm = num_angles*8 #clips per movie

    def __len__(self):
        return (self.nf[-1]-self.frame_mod*self.nf.shape[0])*self.cpm

    def __getitem__(self,idx):
        anf = np.concatenate(([0],(self.nf-np.cumsum(self.frame_mod*np.ones_like(self.nf)))*self.cpm)) #adjusted number of frames per movie
        tmp = np.logical_and(idx>=anf[:-1], idx<anf[1:])
        movie_number = np.nonzero(tmp)[0][0]
        transform_number = (idx-anf[movie_number])//((anf[movie_number+1]-anf[movie_number])/self.cpm)
        transform_number += movie_number*self.cpm
        addon = int(self.frame_mod*transform_number)
        tmp = self.img[idx+addon:idx+addon+self.rnn_length]
        tmp = np.ascontiguousarray(tmp)
        tmp = torch.from_numpy(tmp)
        tmp = tmp.view(self.rnn_length,-1)
        tmp = tmp.float()
        return tmp

class AllImageFolderRNNDataset(Dataset):
    def __init__(self, path, num_frames_list, num_angles=11, rnn_length=4):
        self.path = path
        self.nf = num_frames_list # Should be 1D numpy array of cumulative sum of frames
        self.rnn_length = rnn_length
        self.frame_mod = rnn_length-1
        self.na = num_angles
        self.cpm = num_angles*8 #clips per movie
        
    def __len__(self):
        return (self.nf[-1]-self.frame_mod*self.nf.shape[0])*self.cpm
    
    def __getitem__(self,idx):
        anf = np.concatenate(([0],(self.nf-np.cumsum(self.frame_mod*np.ones_like(self.nf)))*self.cpm)) #adjusted number of frames per movie
        tmp = np.logical_and(idx>=anf[:-1], idx<anf[1:])
        movie_number = np.nonzero(tmp)[0][0]
        transform_number = (idx-anf[movie_number])//((anf[movie_number+1]-anf[movie_number])/self.cpm)
        transform_number += movie_number*self.cpm
        addon = int(self.frame_mod*transform_number)
        img = torch.zeros(self.rnn_length,128,128,dtype=torch.float)
        for i, mov_num in enumerate(range(idx+addon,idx+addon+self.rnn_length)):
            tmp = Image.open(self.path+f'{mov_num:08d}.tif')
            img[i] = torchvision.transforms.ToTensor()(tmp)[0].float()/SCALE_FACTOR
        return img, mov_num

class CroppedImageDataset(Dataset):
    def __init__(self, img):
        self.img = img
        
    def __len__(self):
        return self.img.shape[0]
    
    def __getitem__(self,idx):
        return self.img[idx].cuda()

class SubsetSampler(torch.utils.data.Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices, shuffle=False):
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            return iter(np.random.permutation(self.indices))
        else:
            return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def RandomIndicesForKFoldValidation(ds_len, k, K=5, rand_seed=False, nORp=1):
    """
    k counts from zero up to K-1
    Todo:
        Consider how I handle the random seed and forcing the user to input one.
            I want to allow a random seed, but not require one.
        Consider handling k-counting better
    """
    if rand_seed: np.random.seed(rand_seed)
    rand_idx = np.random.permutation(ds_len)
    if k == 0:
        train_idx = rand_idx[:np.int(ds_len*(K-1)/K)]
        val_idx   = rand_idx[np.int(ds_len*(K-1)/K):]
    elif k == K-1:
        train_idx = rand_idx[np.int(ds_len*1/K):]
        val_idx   = rand_idx[:np.int(ds_len*1/K)]
    else:
        train_idx = np.concatenate((rand_idx[:np.int(ds_len*(K-1-k)/K)], rand_idx[np.int(ds_len*(K-k)/K):]), axis=None)
        val_idx   = rand_idx[np.int(ds_len*(K-1-k)/K):np.int(ds_len*(K-k)/K)]
    if nORp > 1:
        nORp = nORp/ds_len
    return (train_idx[:int(np.round(len(train_idx)*nORp))], val_idx[:int(np.round(len(val_idx)*nORp))])

class kMeansImageRNNDataset(Dataset):
    def __init__(self, img, num_frames_list, num_angles, rnn_length=4):
        self.img = img
        self.nf = num_frames_list # Should be 1D numpy array of cumulative sum of frames
        self.rnn_length = rnn_length
        self.frame_mod = rnn_length-1
        self.na = num_angles+1
        
    def __len__(self):
        return (self.nf[-1]-self.frame_mod*len(self.nf))*self.na
    
    def __getitem__(self,idx):
        anf = np.concatenate(([0],(self.nf-np.cumsum(self.frame_mod*np.ones_like(self.nf)))*self.na)) #adjusted number of frames per movie
        tmp = np.logical_and(idx>=anf[:-1], idx<anf[1:])
        movie_number = np.nonzero(tmp)[0][0]
        transform_number = (idx-anf[movie_number])//((anf[movie_number+1]-anf[movie_number])/self.na)
        transform_number += movie_number*self.na
        addon = int(self.frame_mod*transform_number)
        tmp = self.img[idx+addon:idx+addon+self.rnn_length]
        tmp = np.ascontiguousarray(tmp)
        tmp = torch.from_numpy(tmp)
        tmp = tmp.view(self.rnn_length,32,32)
        # transforms = np.random.rand(3)>.5
        # if transforms[0]:
        #     tmp = tmp.flip(1)
        # if transforms[1]:
        #     tmp = tmp.flip(2)
        # if transforms[2]:
        #     tmp.transpose_(1,2)
        return tmp

class BBImageDataset(Dataset):
    def __init__(self,img,bounds):
        self.img = img
        self.bounds = bounds

    def __len__(self):
        return self.img.shape[0]
    
    def __getitem__(self,idx):
        img = self.img[idx].data.numpy().copy()
        bound = self.bounds[idx].data.numpy().copy()

        cy = bound[0]+bound[2]/2
        cx = bound[1]+bound[3]/2

        y_crop = bound[2]/2 + min([bound[0],512-bound[0]-bound[2]])*.2 + np.random.rand()*min([bound[0],512-bound[0]-bound[2]])*.8
        x_crop = bound[3]/2 + min([bound[1],512-bound[1]-bound[3]])*.2 + np.random.rand()*min([bound[1],512-bound[1]-bound[3]])*.8
        img = img[int(cy-y_crop):int(cy+y_crop),int(cx-x_crop):int(cx+x_crop)]
        y_scale = img.shape[0]/512
        x_scale = img.shape[1]/512
        img = imresize(img,(512,512))

        bound[0] = y_crop-bound[2]/2
        bound[0] /= y_scale
        bound[1] = x_crop-bound[3]/2
        bound[1] /= x_scale
        bound[2] /= y_scale
        bound[3] /= x_scale

        if np.random.rand()>0.5:
            y_shift = np.random.randint(.95*bound[0])
            img[y_shift+1:,:] = img[-1:y_shift:-1,:]
            bound[0] = 512+y_shift-(bound[0]+bound[2])
        else:
            y_shift = np.random.randint(1.05*(bound[0]+bound[2]),512)
            img[:y_shift+1,:] = img[y_shift::-1,:]
            bound[0] = y_shift-(bound[0]+bound[2])
        if np.random.rand()>0.5:
            x_shift = np.random.randint(.95*bound[1])
            img[:,x_shift+1:] = img[:,-1:x_shift:-1]
            bound[1] = 512+x_shift-(bound[1]+bound[3])
        else:
            x_shift = np.random.randint(1.05*(bound[1]+bound[3]),512)
            img[:,:x_shift+1] = img[:,x_shift::-1]
            bound[1] = x_shift-(bound[1]+bound[3])
        
        img = img*(1 + np.random.randn(512,512)/10)*(1 + (np.random.rand()-.5)/10)
        

        return (torch.Tensor(img).view(1,512,512),torch.Tensor(bound))

class PremodifiedBBImageDataset(Dataset):
    def __init__(self,img,bounds):
        self.img = img
        self.bounds = bounds

    def __len__(self):
        return self.img.shape[0]
    
    def __getitem__(self,idx):
        return (self.img[idx].view(1,512,512),self.bounds[idx])

class kMeanAEImageDataset(Dataset):
    def __init__(self,img,k_mean):
        self.img = img
        self.k_mean = k_mean
    
    def __len__(self):
        return self.img.shape[0]
    
    def __getitem__(self,idx):
        return (self.img[idx], self.k_mean[idx])

class BoundingLandmarksDataset(Dataset):
    def __init__(self,img,xy):
        self.img = img
        self.xy = xy
    
    def __len__(self):
        return self.img.shape[0]
    
    def __getitem__(self,idx):
        return (self.img[idx], self.xy[idx])

class Conv_AE_RNN_ImageDataset(Dataset):
    def __init__(self, img, num_frames_list, rnn_length=4):
        self.img = img
        self.nf = np.concatenate(([0],num_frames_list)) if not np.equal(num_frames_list[0],0) else num_frames_list
        self.rnn_length = rnn_length
        self.frame_mod = rnn_length-1
        self.gc = [] #group count
        for i in range(len(self.nf)-1):
            self.gc.append((self.nf[i+1]-self.nf[i])//rnn_length)
        self.gc = np.concatenate(([0],np.cumsum(self.gc)))

    def __len__(self):
        return self.gc[-1]
    
    def __getitem__(self, idx):
        self.get_movie_number(idx)
        addon = int()
        tmp = self.img[idx+addon:idx+addon+self.rnn_length]
        return tmp #tmp.shape == (rnn_length,1,128,128)

    def get_movie_number(self, idx):
        tmp = np.logical_and(idx>=self.gc[:-1], idx<self.gc[1:])
        self.mn = np.nonzero(tmp)[0][0]

class FutureImageDataset(Dataset):
    def __init__(self, img, num_frames_list, input_distance = 8, future_distance = 30):
        self.img = img
        self.nf = num_frames_list
        self.future = future_distance
        self.input = input_distance
        self.frame_mod = future_distance + input_distance - 1
        self.nt = 8 #number of transforms

    def __len__(self):
        return self.nf[-1]-len(self.nf)*self.frame_mod

    def __getitem__(self, idx):
        anf = np.concatenate(([0],(self.nf-np.cumsum(self.frame_mod*np.ones_like(self.nf))))) #adjusted number of frames per movie
        tmp = np.logical_and(idx>=anf[:-1], idx<anf[1:])
        movie_number = np.nonzero(tmp)[0][0]
        addon = int(self.frame_mod*movie_number)
        data_idx = slice(idx+addon,idx+addon+self.input)
        future_idx = idx+addon+self.frame_mod
        out_list = []
        pre_img = self.img[data_idx].clone().detach()
        post_img = self.img[future_idx][None].clone().detach()
        for i in range(8):
            out_pre = pre_img.clone().detach()
            out_post = post_img.clone().detach()
            for j in range(3):
                if (i//2**j)%2 == 1:
                    out_pre = self._transformations[j](self, out_pre)
                    out_post = self._transformations[j](self, out_post)
            out_list.append((out_pre, out_post))
        return out_list
    
    def _flip_ud(self, input):
        return torch.flip(input,[2])
    
    def _flip_lr(self, input):
        return torch.flip(input,[3])
    
    def _T(self, input):
        return input.transpose(2,3)

    _transformations = {0:_flip_ud, 1:_flip_lr, 2:_T}

def random_affine_transform(imgs):
    if not isinstance(imgs,list):
        imgs = [imgs]

    def _flip_ud(input):
        return torch.flip(input,[2])
    def _flip_lr(input):
        return torch.flip(input,[3])
    def _T(input):
        return input.transpose(2,3)

    _transformations = {0:_flip_ud, 1:_flip_lr, 2:_T}

    i = np.random.choice(8)
    for j in range(3):
        if (i//2**j)%2 == 1:
            for i in range(len(imgs)):
                imgs[i] = _transformations[j](imgs[i])
    
    return imgs

def random_crop(imgs, out_size):
    if not isinstance(imgs,list):
        imgs = [imgs]
    in_size = imgs[0].shape[2:4]
    slices = []
    for i in range(2):
        mx = in_size[i] - out_size[i] + 1
        s0 = np.random.choice(mx)
        slices.append(slice(s0,s0+out_size[i]))
    for i in range(len(imgs)):
        imgs[i] = imgs[i][:,:,slices[0],slices[1]]

    return imgs

class EmbeddedFutureImageDataset(Dataset):

    def __init__(self, embedding):
        self.e = embedding

    def __len__(self):
        return self.e.shape[0]
    
    def __getitem__(self, idx):
        return (self.e[idx,:,:-1],self.e[idx,:,-1])

class ActinGanDataset(Dataset):
    def __init__(self, high_exposure, low_exposure, length=None, num_copy=None):
        self.he = high_exposure
        self.le = low_exposure
        self.length = length if length is not None else self.he.shape[0]
        if length is None and num_copy is not None:
            self.length *= num_copy

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        idx = idx%self.he.shape[0]
        imgs = [self.he[idx][None], self.le[idx][None]]
        imgs = random_crop(random_affine_transform(imgs),(128,128))
        for i in range(len(imgs)):
            imgs[i] = (imgs[i]-imgs[i].mean())/imgs[i].std()
        return imgs
        
class ActinClassifierDataset(Dataset):
    def __init__(self, img, classification, length=None):
        self.img = img
        self.cls = classification
        self.length = length if length is not None else self.img.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx%self.img.shape[0]
        img = self.img[idx]
        return (random_affine_transform(img[None])[0][0], self.cls[idx])

class ActinUNetDataset(Dataset):
    def __init__(self, high_low):
        self.he, self.le = high_low

    def __len__(self):
        return self.he.shape[0]

    def __getitem__(self, idx):
        imgs = random_affine_transform([self.le[idx][None], self.he[idx][None]])
        return (imgs[0][0], imgs[1][0])