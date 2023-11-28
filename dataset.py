import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

def load_tensor(image, resize_h=256, resize_w=256, mean = -500, std = 500):
    u_water = 0.0192
    image = (image - u_water) * 1000 / u_water
    image = cv2.resize(image, (resize_h, resize_w)).astype('float32')
    image = torch.tensor(image)
    image = (image - mean) / std
    image = image.view(1, resize_h, resize_w) 
    return image


class npyDataset(Dataset):
    def __init__(self, npy_file, resize_h, resize_w):
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.data = dict(np.load(npy_file, allow_pickle=True).tolist())
        self.keys = list(self.data.keys())
        for key in self.keys:
            self.data[key] = np.transpose(self.data[key], [2, 0, 1]) # to (D, H, W)
            self.data[key] = self.data[key][:, np.newaxis, :, :] # to (D, 1, H, W)

    def __len__(self):
        return len(self.data[self.keys[0]])
    
    def __getitem__(self, idx):
        f_nd_data = self.data['f_nd'][idx,0,:,:] # (H, W)
        f_qd_data = self.data['f_qd'][idx,0,:,:]  

        f_nd_data = load_tensor(f_nd_data, self.resize_h, self.resize_w) # (1, re_H, re_W)
        f_qd_data = load_tensor(f_qd_data, self.resize_h, self.resize_w)

        return f_nd_data, f_qd_data
    

if __name__=="__main__":
    npy_file_path = 'data/testset.npy'
    npy_dataset = npyDataset(npy_file=npy_file_path, resize_h=256, resize_w=256)
    dataloader = DataLoader(npy_dataset, batch_size=64, shuffle=False)

    print(len(npy_dataset))

    for batch in dataloader:
        f_nd_batch, f_qd_batch = batch
        print(f_nd_batch.shape, f_qd_batch.shape)