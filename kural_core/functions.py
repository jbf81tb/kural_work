import numpy as np
from line_profiler import LineProfiler

def k_means_classifier(img,K=3):
    if K>255:
        print('K must be less than 2^8-1.\nSetting K to 255')
        K=255
    cent = np.asarray([np.percentile(img.flatten(),99.999*i/(K-1)) for i in range(K)]).reshape(1,K)
    nf = img.shape[0]
    ppf = img.shape[1]*img.shape[2] #pixels per frame
    limit = 1260/K
    cycles = np.ceil(nf/limit).astype(int)
    frame_list = [i*nf//cycles for i in range(cycles)]
    frame_list.append(nf)
    print('Finished iteration ',end='')
    print_str = '0'
    print(print_str,end='')
    for iteration in range(15):
        nearest_cent = np.zeros(img.size,dtype=np.uint8)
        for i in range(len(frame_list)-1):
            st = frame_list[i]
            end = frame_list[i+1]
            nearest_cent[st*ppf:end*ppf] = np.argmin(np.abs(img[st:end].reshape(-1,1)-cent),axis=1)
        for i in range(K):
            cent[0,i] = np.mean(img.reshape(-1,1)[nearest_cent==i])
        for _ in range(len(print_str)): print('\b',end='')
        print_str = f'{iteration}'
        print(print_str,end='')
    nearest_cent = np.zeros(img.size,dtype=np.uint8)
    for i in range(len(frame_list)-1):
        st = frame_list[i]
        end = frame_list[i+1]
        nearest_cent[st*ppf:end*ppf] = np.argmin(np.abs(img[st:end].reshape(-1,1)-cent),axis=1)
    print('\nDone')
    return (nearest_cent.reshape(*img.shape), cent)

def do_profile(follow=[]):
    def inner(func):
        def profiled_func(*args,**kwargs):
            try:
                profiler = LineProfiler()
                profiler.add_function(func)
                for f in follow:
                    profiler.add_module(f)
                profiler.enable_by_count()
                return func(*args,**kwargs)
            finally:
                profiler.print_stats()
        return profiled_func
    return inner