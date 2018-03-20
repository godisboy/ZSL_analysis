import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from models.VAE import VAE_C
from data.data_process import ImagelistDataset, Get_File_List
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import argparse
import numpy as np

from utils.KNN_ZSL import compute_W, knn_zsl_el, NormaliseRows

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='checkpoints/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

### loading dataset ###
get_file_list = Get_File_List()
files_train, files_test, S_train, S_test, Y_test, test_classes, attritubes = get_file_list.get_list()
cub_train = ImagelistDataset(files_train, S_train,
                             transform=transforms.Compose([
                                 transforms.Resize(64),
                                 transforms.CenterCrop(64),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
                             )
cub_test = ImagelistDataset(files_test, S_test,
                             transform=transforms.Compose([
                                 transforms.Resize(64),
                                 transforms.CenterCrop(64),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
                             )

train_loader = DataLoader(cub_train, batch_size=16, shuffle=True, num_workers=int(2))
test_loader = DataLoader(cub_test, batch_size=1, shuffle=False, num_workers=int(2))
eval_train_loader = DataLoader(cub_train, batch_size=1, shuffle=False, num_workers=int(2))
print(len(train_loader))

vae = VAE_C()
optimizer = optim.Adam(vae.parameters(), lr=0.0001)

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, size_average=False)
    KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    BCE /= x.size(0)
    KLD /= x.size(0)
    return BCE, KLD, BCE + 1. * KLD

if opt.cuda:
    vae.cuda()

def test(model):
    train_feature = torch.FloatTensor(8821, 128)
    for i, (data, _) in enumerate(eval_train_loader):
            input = Variable(data.cuda())
            mu, logvar = model.encode(input)
            z = model.reparameterize(mu, logvar)
            train_feature[i, :] = z.cpu().data

    test_feature = torch.FloatTensor(2967, 128)
    for j, (data, _) in enumerate(test_loader):
            input = Variable(data.cuda())
            mu, logvar = model.encode(input)
            z = model.reparameterize(mu, logvar)
            test_feature[j, :] = z.cpu().data

    W = compute_W(train_feature.numpy(), S_train)
    S_est = np.matmul((test_feature), W)

    print(W.shape, test_feature.shape,  S_est.shape)
    S_test_proto = []
    test_class_id = []
    for a_i in test_classes:
        a_i_ind = int(a_i[0][0:3]) - 1
        S_test_proto.append(attritubes[a_i_ind, :])
        test_class_id.append(a_i_ind)
    Acc, Y_est = knn_zsl_el(NormaliseRows(S_est), np.array(S_test_proto), np.array(Y_test), np.array(test_class_id))
    print('ZSL Accuracy: {:.2f}%'.format(Acc))

for epoch in range(opt.niter):
    for i, (data, _) in enumerate(train_loader):
        vae.zero_grad()
        input = Variable(data.cuda())
        out, mu, logvar = vae(input)
        BCE, KLD, loss = loss_fn(out, input, mu, logvar)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print('[%d/%d][%d/%d], loss: %f' % (epoch, opt.niter, i, len(train_loader), loss.data[0]))
        if i % 200 == 0:
            vutils.save_image(data,
                              '%s/real_samples.png' % opt.outf,
                              normalize=True)
            vutils.save_image(out.data,
                              '%s/rec_samples_epoch_%03d.png' % (opt.outf, epoch),
                              normalize=True)
    if epoch % 5 == 0:
        test(vae)




