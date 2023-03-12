import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import warnings
warnings.filterwarnings("ignore")

# download the dataset

img_size = 64
batch_size = 64

lr = 0.002
beta = 0.5
n_iter = 25
outf = "output"

dataset = torchvision.datasets.CIFAR10(
  root = 'data',
  download = True,
  transform = transforms.Compose([
    transforms.Resize(img_size), transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))
  
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)

print("data loaded.")

# initialize size of the vector
nz = 100

# intitialize filter size of the generator
ngf = 64

# intitialize filter size of the discriminator
ndf = 64

# output image channels
nc = 3

# custom weight initialiization calledn on netG and netD

def weight_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv')!= -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm')!= -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)
  elif classname.find('Linear')!= -1:
    m.weight.data.normal_(0.0, 0.02)
  # else:
  #   raise Exception("Unknown weight initializer: " + classname)

# generator class

class NetG(nn.Module):
  def __init__(self):
    super().__init__()

    self.model = nn.Sequential(
      nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
      nn.BatchNorm2d(ngf * 8),
      nn.ReLU(True),
      nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 4),
      nn.ReLU(True),
      nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 2),
      nn.ReLU(True),
      nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf),
      nn.ReLU(True),
      nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias = False),
      nn.Tanh()
    )

  def forward(self, input):
    output = self.model(input)
    return output

netG = NetG()
netG.apply(weight_init)
print(netG)

# discriminator class

class NetD(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
      nn.Conv2d(nc, ndf, 4, 2, 1, bias = False),
      nn.LeakyReLU(0.2, inplace = True),
      nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias = False),
      nn.BatchNorm2d(ndf * 2),
      nn.LeakyReLU(0.2, inplace = True),
      nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias = False),
      nn.BatchNorm2d(ndf * 4),
      nn.LeakyReLU(0.2, inplace = True),
      nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias = False),
      nn.BatchNorm2d(ndf * 8),
      nn.LeakyReLU(0.2, inplace = True),
      nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias = False),
      nn.Sigmoid()
    )

  def forward(self, input):
    output = self.model(input)
    return output.view(-1, 1).squeeze(1)

netD = NetD()
netD.apply(weight_init)
print(netD)

# defining the loss function and optimizer

criterion = nn.BCELoss()
input = torch.FloatTensor(batch_size, 3, img_size, img_size)
noise = torch.FloatTensor(batch_size, nz, 1, 1)
fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1).norm(2, 2).normal_(0, 1)
label = torch.FloatTensor(batch_size)
real_label = 1
fake_label = 1

fixed_noise = Variable(fixed_noise)

optimizer_d = optim.Adam(params = netD.parameters(), lr = lr)
optimizer_g = optim.Adam(params = netG.parameters(), lr = lr)


for epoch in range(n_iter):

  for i, data in enumerate(dataloader, 0):
    # update discriminator network
    netD.zero_grad()
    real_, _ = data
    batch_size = real_.size(0)
    input.resize_as_(real_).copy_(real_)
    label.resize_(batch_size).fill_(real_label)
    inputv = Variable(input)
    labelv = Variable(label)
    output = netD(inputv)
    error_D = criterion(output, labelv)
    error_D.backward()
    D_ = output.data.mean()

    # train with fake image
    noise.resize(batch_size, nz, 1, 1).normal_(0, 1)
    noisev = Variable(noise)
    fake = netG(noisev)
    labelv = Variable(label.fill_(fake_label))
    output = netD(fake.detach())
    error_D_fake = criterion(output, labelv)
    error_D_fake.backward()
    D_z1 = output.data.mean()
    error_D_total = error_D + error_D_fake 
    optimizer_d.step()

    # update generator network
    netG.zero_grad()
    labelv = Variable(label.fill_(real_label))
    output = netD(fake)
    error_G = criterion(output, labelv)
    error_G.backward()
    D_G_z2 = output.data.mean()
    optimizer_g.step()


    vutils.save_image(real_, "real_sample.jpg", normalize = True)
    fake = netG(fixed_noise)
    vutils.save_image(fake.data, "fake_samples_epoch_{}.jpg".format(epoch), normalize = True)
                      
print("done.")
    
