import torch
import torch.nn as nn
import torch.nn.functional as F

class NFrameLinearModel(nn.Module):
    def __init__(self, num_input_frames=3):
        super().__init__()
        self.intra_relate = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
        )
        self.inter_relate = nn.Sequential(
            nn.Linear(2*256, 2*256),
            nn.ReLU(inplace=True),
            nn.Linear(2*256, 2*256),
            nn.ReLU(inplace=True),
            nn.Linear(2*256, 2*256),
            nn.ReLU(inplace=True),
            nn.Linear(2*256, 2*256),
            nn.ReLU(inplace=True),
            nn.Linear(2*256, 256)
        )
        self.nf = num_input_frames
        
    def forward(self, img):
        bs = img.shape[0]
        for i in range(self.nf):
            coding = self.intra_relate(img[:,i,:])
            if i==0: h = coding
            h = self.inter_relate(torch.cat((h,coding),dim=1))
        return h

class NFrameConvolutionalModel(nn.Module):
    def __init__(self, num_input_frames=3):
        super().__init__()
        self.compress = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(1,16,7),
            nn.InstanceNorm2d(16),
            # nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.ReflectionPad2d(3),
            nn.Conv2d(16,32,7),
            nn.InstanceNorm2d(32),
            # nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.ReflectionPad2d(3),
            nn.Conv2d(32,64,7),
            nn.InstanceNorm2d(64),
            # nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64,1,1)
        )
        self.evolve = nn.Sequential(
            nn.Conv2d(num_input_frames,1000,1),
            nn.Conv2d(1000,1,1)
        )
        self.decompress = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.ConvTranspose2d(1,16,4,stride=2,padding=3),
            nn.InstanceNorm2d(16),
            # nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.ConvTranspose2d(16,32,4,stride=2,padding=3),
            nn.InstanceNorm2d(32),
            # nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.ConvTranspose2d(32,64,4,stride=2,padding=3),
            nn.InstanceNorm2d(64),
            # nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64,1,7)
        )
        self.nf = num_input_frames

        
    def forward(self, img):
        bs = img.shape[0]
        compressed = torch.zeros((bs,self.nf,16,16),device=torch.device('cuda'))
        for i in range(self.nf):
            compressed[:,i,:,:] = self.compress(img[:,i,:,:].view(bs,1,128,128)).view(bs,16,16)
        x = self.evolve(compressed)
        return self.decompress(x).view(bs,128,128)


class MaxPoolDecoderModel(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.mp = nn.MaxPool2d(kernel_size)
        self.fc1 = nn.Linear(8**2,16**2)
        self.fc2 = nn.Linear(16**2,32**2)
        self.fc3 = nn.Linear(32**2,64**2)
        self.do1 = nn.Dropout(p=0.001)
        self.do2 = nn.Dropout(p=0.02)
        
    def forward(self, x):
        x = self.mp(x).view(-1,8**2)
        x = F.relu(self.fc1(x))
        x = self.do1(x)
        x = F.relu(self.fc2(x))
        x = self.do2(x)
        x = torch.clamp(self.fc3(x),0,1)
        return x

class kMeansModel(nn.Module):
    def __init__(self,num_input_frames=3):
        super().__init__()
        self.inter_relate = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024,1024),
            nn.ReLU(True)
        )
        self.hidden = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU(True)
        )
        self.evolve = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(1,512,7),
            nn.BatchNorm2d(512),
            nn.ReflectionPad2d(3),
            nn.Conv2d(512,1,7),
            nn.Hardtanh(0,2,inplace=True)
        )
    def forward(self,img):
        bs = img.shape[0]
        nf = img.shape[1]
        for i in range(nf):
            coding = self.inter_relate(img[:,i,:,:].view(bs,1024))
            if i == 0: 
                h = self.hidden(coding)
            else:
                h = self.hidden(h+coding)
        x = self.evolve(h.view(bs,1,32,32))
        return x

class BoundingBoxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(1,32,9,5),
            nn.InstanceNorm2d(32,affine=True),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(32,64,5,3),
            nn.InstanceNorm2d(64,affine=True),
            nn.ReLU(True)
        )
        self.regression = nn.Sequential(
            nn.Linear(64*7*7,64*7),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64*7,64*7),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64*7,4),
            nn.Sigmoid()
        )
    def forward(self,img):
        x = self.convolution(img)
        x = self.regression(x.view(-1,64*7*7))
        return 512*x

class IoULoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,a,b):
        dh = torch.min(torch.cat([a[:,0:1]+a[:,2:3],b[:,0:1]+b[:,2:3]],dim=1),dim=1)[0]-\
             torch.max(torch.cat([a[:,0:1],b[:,0:1]],dim=1),dim=1)[0]
        dw = torch.min(torch.cat([a[:,1:2]+a[:,3:],b[:,1:2]+b[:,3:]],dim=1),dim=1)[0]-\
             torch.max(torch.cat([a[:,1:2],b[:,1:2]],dim=1),dim=1)[0]
        intersect = dh*dw*(dh>0).float()*(dw>0).float()
        
        union = a[:,2]*a[:,3] + b[:,2]*b[:,3] - intersect

        return torch.sum(1-intersect/union)/union.shape[0]

class kMeansAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(64*64,64*16),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64*16,64*4),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64*4,64*1),
            nn.ReLU(True)
        )
        self.decode = nn.Sequential(
            nn.Linear(64*1,64*8),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64*8,64*64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64*64,64*64*2),
            nn.ReLU(True)
        )
    def forward(self,img):
            x = self.encode(img.view(-1,64*64))
            x = self.decode(x)
            return x.view(-1,2,64,64)

class BoundingPointsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1,8,3,1),
            nn.InstanceNorm2d(8),
            nn.ReLU(True),
            nn.MaxPool2d(2),#128->64

            nn.ReflectionPad2d(1),
            nn.Conv2d(8,16,3,1),
            nn.InstanceNorm2d(16),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(16,16,3,1),
            nn.InstanceNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2),#64->32

            nn.ReflectionPad2d(1),
            nn.Conv2d(16,32,3,1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32,32,3,1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2),#32->16

            nn.ReflectionPad2d(1),
            nn.Conv2d(32,64,3,1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64,64,3,1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),#16->8

            nn.ReflectionPad2d(1),
            nn.Conv2d(64,128,3,1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128,128,3,1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2)#8->4
        )
        self.regression = nn.Sequential(
            nn.Linear(128*4*4,128*4*4),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(128*4*4,512),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(512,512),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(512,512),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(512,30)
        )
    def forward(self,img):
        x = self.convolution(img)
        x = self.regression(x.view(-1,128*4*4))
        return x

class ConvolutionalAutoencoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1,8,4,2), #128->64
            nn.InstanceNorm2d(8),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8,16,4,2), #64->32
            nn.InstanceNorm2d(16),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(16,16,4,2), #32->16
            nn.InstanceNorm2d(16),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(16,32,4,2), #16->8
            nn.InstanceNorm2d(32),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(32,32,4,2), #8->4
            nn.InstanceNorm2d(32),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(32,64,4,2), #4->2
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64,128,2), #2->1
        )

        self.deconvolution = nn.Sequential(
            nn.ConvTranspose2d(128,128,2), #1->2
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128,128,4,2,1), #2->4
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128,64,4,2,1), #4->8
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64,64,4,2,1), #8->16
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64,64,4,2,1), #16->32
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64,64,4,2,1), #32->64
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64,64,4,2,1), #64->128
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64,32,3,1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32,32,3,1,1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32,16,3,1),
            nn.InstanceNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16,16,3,1,1),
            nn.InstanceNorm2d(16),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(16,1,3,1)
        )
    
    def forward(self,img):
        x = self.convolution(img) #x.shape == (batch,128,1,1)
        return self.deconvolution(x) #shape == (batch,1,128,128)

class Conv_AE_RNN_Model(nn.Module):
    def __init__(self,n_hidden,n_layers,bs):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.bs = bs
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True
            )
        self.reshape_layer = nn.Sequential(
            nn.Dropout(p=.3),
            nn.Linear(n_hidden,n_hidden),
            nn.ReLU(True),
            nn.Dropout(p=.3),
            nn.Linear(n_hidden,int(n_hidden//2)),
            nn.ReLU(True),
            nn.Dropout(p=.3),
            nn.Linear(int(n_hidden//2),128)
        )
        self.init_hidden()
        self.init_cell()

    def forward(self,input):
        output, (hidden, cell) = self.lstm(input, (self.h, self.c))
        self.h = hidden.clone().detach().requires_grad_(True)
        self.c = cell.clone().detach().requires_grad_(True)
        out = self.reshape_layer(output)
        return out

    def init_hidden(self):
        self.h = torch.randn((self.n_layers,self.bs,self.n_hidden),device=torch.device('cuda'))
        self.h /= torch.sum(self.h)
        self.h.requires_grad_(True)

    def init_cell(self):
        self.c = torch.randn((self.n_layers,self.bs,self.n_hidden),device=torch.device('cuda'))
        self.c /= torch.sum(self.c)
        self.c.requires_grad_(True)

class FuturePrediction1DConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential( #in shape = (batch,128,8)
            nn.Conv1d(128,128,3,1,1),
            nn.ReLU(True),
            nn.Conv1d(128,128,3,1,1),
            nn.ReLU(True),
            
            nn.Conv1d(128,256,4,2,1), #8->4
            nn.ReLU(True),
            nn.Conv1d(256,256,3,1,1),
            nn.ReLU(True),

            nn.Conv1d(256,512,4,2,1), #4->2
            nn.ReLU(True),
            nn.Conv1d(512,512,3,1,1),
            nn.ReLU(True),

            nn.Conv1d(512,1024,2), #2->1 
        ) #out shape = (batch,1024,1)
        self.lin = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU(True),
            nn.Dropout(p=0.1),
            nn.Linear(1024,512),
            nn.ReLU(True),
            nn.Dropout(p=0.1),
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Dropout(p=0.1),
            nn.Linear(256,128)
        )

    def forward(self,input):
        x = self.conv(input)
        x = self.lin(x.squeeze(2))
        return x
