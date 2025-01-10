import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import torch
import numpy as np
import argparse

from options.options import parse  
from PIL import Image
from glob import glob
from model.flol import *
from ptflops import get_model_complexity_info
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MyDataset(Dataset):
    """
    A Dataloader has always the same structure and 3 parts:
    1) __ini__ to create the Dataloader instance
    2) __len__ to access the length of the dataloader: how many images?
    3) __getitem__ yields whatever we need to feed into the model, and the corresponding ground-truth.
    
    We can add more attributes and functions here,
    """
    
    def __init__(self, images_list_low, images_list_high, test=False):
        """
        - images_list: list of RGB images used for training or testing the model
        - test: indicates if the dataset is for training (False) or testing (True)
        """
        self.imgs_low = sorted(images_list_low)
        self.imgs_high = sorted(images_list_high)
        self.test = test
        
    def __len__(self):
        return len(self.imgs_low)

    def __getitem__(self, idx):
        """
        Given a (random) index. The dataloader selects the corresponding image path, and loads the image.
        Then it returns the image, after applying any required transformation.
        """
        img_path_low = self.imgs_low[idx]

        # Load the low image
        rgb_low = np.array(Image.open(img_path_low).convert('RGB'))
        
        # Normalize to [0,1] fp32
        rgb_low = (rgb_low / 255).astype(np.float32)
        assert rgb_low.shape[-1] == 3 # check if it is a 3ch image
        img_path_high = self.imgs_high[idx]

        # Load the high image
        rgb_high = np.array(Image.open(img_path_high).convert('RGB'))
        # Normalize to [0,1] fp32
        rgb_high = (rgb_high / 255).astype(np.float32)
        assert rgb_high.shape[-1] == 3 # check if it is a 3ch image
        
        # Convert to PyTorch
        rgb_low = torch.from_numpy(rgb_low)
        rgb_high = torch.from_numpy(rgb_high)

        # Change to Pytorch format, instead of [H,W,3] to [3,H,W]
        rgb_low = rgb_low.permute((2, 0, 1))
        rgb_high = rgb_high.permute((2, 0, 1))

        
        return rgb_low, rgb_high
    


def main(weights_path, path_low, path_high, extension):

    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    print("GPU visible devices: " + str(torch.cuda.device_count()))

    PATH_LOW_TEST = path_low
    PATH_HIGH_TEST = path_high

    IMGS_PATHS_LOW_TEST = glob(os.path.join(PATH_LOW_TEST, extension))
    IMGS_PATHS_HIGH_TEST = glob(os.path.join(PATH_HIGH_TEST, extension))

    BATCH_SIZE_TEST = 1

    test_dataset = MyDataset(IMGS_PATHS_LOW_TEST, IMGS_PATHS_HIGH_TEST, test=True)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=1,
                            pin_memory=True, drop_last=True)

    model = create_model()
    checkpoint = weights_path
    checkpoint = torch.load(checkpoint, map_location='cuda:0')
    model.load_state_dict(checkpoint['params']) 
    model.to(device)

    macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, backend='pytorch',
                                            print_per_layer_stat=True, verbose=True)

    print(macs, params)

    model.eval()

    valid_loss_list = list()
    valid_psnr_list = list()
    valid_ssim_list = list()

    #Now we need to go over the test_loader and evaluate the results of the epoch
    for low_batch_valid, high_batch_valid in test_loader:
        
        high_batch_valid = high_batch_valid.to(device)
        low_batch_valid = low_batch_valid.to(device)

        with torch.no_grad():

            enhanced_image = model(low_batch_valid)
            # loss
            valid_loss_batch     = torch.mean((high_batch_valid - enhanced_image)**2)
            # PSNR (dB) metric
            valid_psnr_batch     = 20 * torch.log10(torch.max(high_batch_valid) / torch.sqrt(valid_loss_batch))

            # SSIM (dB) metric
            ssim = SSIM(data_range=1.0).to(device)
            valid_ssim_batch = ssim(enhanced_image, high_batch_valid)

        valid_loss_list.append(valid_loss_batch.item())
        valid_psnr_list.append(valid_psnr_batch.item())
        valid_ssim_list.append(valid_ssim_batch.item())

    print(f"PSNR_Validation:{np.mean(valid_psnr_list)}\t SSIM_Validation:{np.mean(valid_ssim_list)}\t")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./options/UHDLL.yml")

    args = parser.parse_args()
    path_options = args.config
    opt = parse(path_options)

    PATH_WEIGHT = opt['settings']['weight']
    EXTENSION = opt['settings']['extension_images']
    PATH_LOW = opt['paths_images']['low']
    PATH_HIGH = opt['paths_images']['high']

    main(weights_path=PATH_WEIGHT, path_low=PATH_LOW, path_high=PATH_HIGH, extension = EXTENSION)

