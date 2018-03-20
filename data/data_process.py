import os
import csv
import numpy as np
from scipy.misc import imresize
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from PIL import Image
##### images path######
class Get_File_List(object):
    def __init__(self):
        self.num = 0
    def get_list(self):
        path = '/home/chao/zero/datasets/CUB_200_2011/CUB_200_2011/images'
        files = [os.path.join(path, filei) for filei in os.listdir(path)]

        images_list = []
        with open('/home/chao/zero/datasets/CUB_200_2011/CUB_200_2011/images.txt') as inputfile:
            for row in csv.reader(inputfile):
                images_list.append(row[0].split(' ')[1])
        images_list = np.array(images_list)

        attritubes_t = []
        with open('/home/chao/zero/datasets/CUB_200_2011/CUB_200_2011/attributes/class_attribute_labels_continuous.txt') as inputfile:
            for row in csv.reader(inputfile):
                attritubes_t.append(row[0])
        # numpy string
        attritubes_t = np.array(attritubes_t)

        attritubes = np.zeros((200, 312))
        for i in range (200):
            str1 = attritubes_t[i].split(' ')
            for j, data in enumerate(str1):
                attritubes[i, j] = data

        train_classes = []
        with open('/home/chao/zero/datasets/xlsa17/data/CUB/trainvalclasses.txt') as inputfile:
            for row in csv.reader(inputfile):
                train_classes.append(row)
        train_classes = np.array(train_classes)

        test_classes = []
        with open('/home/chao/zero/datasets/xlsa17/data/CUB/testclasses.txt') as inputfile:
            for row in csv.reader(inputfile):
                test_classes.append(row)
        test_classes = np.array(test_classes)

        ### SEEN classes
        N = 7057+1764
        k = 2048
        d = 312
        c = 150

        S_train = np.zeros((N, d)) # Attribute
        files_train = []
        index_counter_train = 0                # Counter

        for i in range(c): #150
            #print(index_counter_train)
            for j in range(images_list.shape[0]):

                if (images_list[j].split('/')[0] == train_classes[i][0].split(' ')[0]):
                    files_train.append(os.path.join('/home/chao/zero/datasets/CUB_200_2011/CUB_200_2011/images/', images_list[j]))
                    S_train[index_counter_train, :] = attritubes[int(images_list[j][0:3])-1, :]
                    index_counter_train += 1

        print(len(files_train), len(S_train))
        # Unseen classes
        N = 2967 # the number of images for seen classes
        k = 2048 # feature dimension
        d = 312  # attribute dimension
        c = 50   # the number of unseen classes

        S_test = np.zeros((N, d)) # Attribute
        files_test = []
        Y_test = []
        index_counter_test = 0                # Counter

        for i in range(c):
            #print(index_counter_test)
            for j in range(images_list.shape[0]):
                if (images_list[j].split('/')[0] == test_classes[i]):
                    files_test.append(os.path.join('/home/chao/zero/datasets/CUB_200_2011/CUB_200_2011/images/', images_list[j]))
                    Y_test.append(int(images_list[j][0:3])-1)
                    S_test[index_counter_test, :] = attritubes[int(images_list[j][0:3])-1, :]
                    index_counter_test += 1
        return files_train, files_test, S_train, S_test, Y_test, test_classes, attritubes



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    """import accimage
    try:
        return accimage.Image(path)
    except IOError:
    """    # Potentially a decoding problem, fall back to PIL.Image
    return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
class ImagelistDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_list, attritubes, transform=None, loader=default_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_list = img_list
        self.attr_list = attritubes
        #self.root_dir = root_dir
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        img = self.loader(img_path)
        if self.transform is not None:
            sample = self.transform(img)
        attritubes = self.attr_list[idx]
        return sample, attritubes

"""
cub_train = ImagelistDataset(files_train, S_train,
                             transform=transforms.Compose([
                                 transforms.Resize(128),
                                 transforms.CenterCrop(128),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
                             )

print(len(cub_train))

dataloader = DataLoader(cub_train, batch_size=16, shuffle=False, num_workers=int(2))
print(len(dataloader))

for data, attrs in dataloader:
    print(data.size(), attrs)
    break
"""


