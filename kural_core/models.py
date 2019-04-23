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

class res_block(nn.Module):
    def __init__(self, c0, do_batch=[True, True], do_ReLU = [True, True]):
        super().__init__()
        self.conv = []
        self.conv.append(nn.Conv2d(c0,c0,3,1,1))
        if do_batch[0]: self.conv.append(nn.BatchNorm2d(c0))
        if do_ReLU[0]:  self.conv.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
        self.conv.append(nn.Conv2d(c0,c0,3,1,1))
        if do_batch[1]: self.conv.append(nn.BatchNorm2d(c0))
        if do_ReLU[1]:  self.conv.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
        self.conv = nn.Sequential(*self.conv)

    def forward(self,x):
        return x + self.conv(x)*0.01

class ActinClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,8,3,1,1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=0.01,inplace=True),

            self._down_block(8,16), #64
            res_block(16),
            self._down_block(16,32), #32
            res_block(32),
            self._down_block(32,64), #16
            res_block(64),
            self._down_block(64,128), #8
            res_block(128),

            nn.Conv2d(128,128,4,2,1), #4
        )
        self.classify = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128*4*4,128),
            nn.LeakyReLU(negative_slope=0.01,inplace=True),
            nn.Dropout(),
            nn.Linear(128,16),
            nn.LeakyReLU(negative_slope=0.01,inplace=True),
            nn.Linear(16,2)
        )
        self._initialize_weights()

    def _down_block(self,c0,c1):
        return nn.Sequential(
            nn.Conv2d(c0,c1,4,2,1),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(negative_slope=0.01,inplace=True)
        )

    def forward(self,x):
        x = self.features(x).view(-1,128*4*4)
        return self.classify(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ActinUNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = inconv(1, 32) #128
        self.down1 = down(32, 64) #64
        self.down2 = down(64, 128) #32
        self.up1 = up(128, 64) #32
        self.up2 = up(64, 32) #64
        self.out_mean = outconv(32, 1) #128
        self.out_std = outconv(32, 1) #128
        self._initialize_weights()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        y = self.up1(x3, x2)
        y = self.up2(y, x1)
        y_mean = self.out_mean(y)
        y_std = self.out_std(y)
        return (x + y_mean, y_std)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.01)
                nn.init.constant_(m.bias, 0)


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            res_block(out_ch, do_batch=[False, False], do_ReLU=[True, False])
        )

    def forward(self, x): return self.conv(x)


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(
            res_block(in_ch, do_batch=[False, False], do_ReLU=[True, False]),
            nn.Conv2d(in_ch, out_ch, 4, 2, 1),
            nn.LeakyReLU(negative_slope=0.01,inplace=True)
        )

    def forward(self, x): return self.mpconv(x)


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = nn.Sequential(
            res_block(in_ch, do_batch=[False, False], do_ReLU=[True, False]),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        )


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Sequential(
            res_block(in_ch, do_batch=[False, False], do_ReLU=[True, False]),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        )

    def forward(self, x): return self.conv(x)

class ActinProbabalisticLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predicted, actual):
        actual = actual.flatten()
        means = predicted[0].flatten()
        stds = nn.Softmax(0)(predicted[1].flatten()) + 0.001
        loss = torch.mean(torch.abs(means-actual)/stds + torch.log(stds))
        return loss

class ActinUNetPerceptualLoss(nn.Module):
    def __init__(self, classifier_model):
        super().__init__()
        self.m = classifier_model
        blocks = []
        lmm = list(iter(self.m.modules()))
        for i, m in enumerate(lmm):
            if hasattr(m, 'stride'):
                if m.stride == (2,2):
                    for j in range(i-1,-1,-1):
                        if isinstance(lmm[j],nn.Conv2d):
                            blocks.append(j)
                            break
            if len(blocks)==3: break
        self.blocks = blocks

    def forward(self, y_pred, y_actual):
        out_pred = [y_pred]
        out_actual = [y_actual]
        start = True
        for b in self.blocks:
            for j, m in enumerate(self.m.modules()):
                if not (isinstance(m,nn.Conv2d) or isinstance(m,nn.BatchNorm2d) or isinstance(m, nn.LeakyReLU)): continue
                if start:
                    out = m(out_pred[0])
                    start = False
                else:
                    out = m(out)
                if j == b: break
            out_pred.append(out*0.1)
            start = True
        for b in self.blocks:
            for j, m in enumerate(self.m.modules()):
                if not (isinstance(m,nn.Conv2d) or isinstance(m,nn.BatchNorm2d) or isinstance(m, nn.LeakyReLU)): continue
                if start:
                    out = m(out_actual[0])
                    start = False
                else:
                    out = m(out)
                if j == b: break
            out_actual.append(out*0.1)
            start = True
        loss = 0
        for p, a in zip(out_pred, out_actual):
            loss += nn.L1Loss()(p,a)
        return loss
