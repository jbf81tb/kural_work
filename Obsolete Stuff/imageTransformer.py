def imageTransformer(imgs:'3D np array of images with shape (n_frames,side,side)', 
                     steps:'Number of steps in RNN + 1 for output' = 4,
                     sz:'Num of px per side of square crop'        = 32):
    '''
    Take an array of images and output cropped versions for use in image RNN.
    '''
    frames, *shape = imgs.shape
    start = np.random.choice(range(frames-steps))
    left = np.random.choice(range(shape[1]-sz))
    bottom = np.random.choice(range(shape[0]-sz))
    mirror = np.random.rand()>.5
    turns = np.random.choice(range(4))
    out_img = []
    for i in range(start,start+steps):
        tmp_img = imgs[i,bottom:bottom+sz,left:left+sz]
        if mirror: tmp_img = tmp_img.T
        tmp_img = np.rot90(tmp_img,k=turns)
        out_img.append(tmp_img)
    return(out_img)