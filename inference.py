import torch, os
import torch.nn.functional as F
import torchvision.models
import argparse

from model.flol import create_model
from options.options import parse 
from torchvision.transforms import Resize
from PIL import Image

pil_to_tensor = torchvision.transforms.ToTensor()
tensor_to_pil = torchvision.transforms.ToPILImage()

def save_tensor(tensor, path):
    
    tensor = tensor.squeeze(0)
    print(tensor.shape, tensor.dtype, torch.max(tensor), torch.min(tensor))
    img = tensor_to_pil(tensor)
    img.save(path)

def pad_tensor(tensor, multiple = 8):
    '''pad the tensor to be multiple of some number'''
    multiple = multiple
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value = 0)
    
    return tensor

def tensorimg2npimg (img):
    """
    We can convert a tensor pytorch Image in [3,H,W] format into a standard np array
    .detach().cpu() moves the array back to the CPU
    .permute() changes to the format [H,W,3]
    .numpy() converts the array to np
    """
    return img.detach().cpu().permute(1, 2, 0).numpy()

def path_to_tensor(path):
    img = Image.open(path).convert('RGB')
    img = pil_to_tensor(img).unsqueeze(0)    
    return img

def main(weights_path, path_images, save_path, resize = True):

    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

    model = create_model()
    checkpoint = weights_path
    checkpoint = torch.load(checkpoint, map_location='cuda:0')
    model.load_state_dict(checkpoint['params'])
    model.to(device)

    path = path_images + '/'
    for image in os.listdir(path):

        img = path_to_tensor(path+image)
        _, _, H, W = img.shape
        img = img.to('cuda')
        if resize:
            new_size = [int(dim) for dim in (H, W)]
            downsample = Resize(new_size)
        else:
            downsample = torch.nn.Identity()

        tensor = downsample(img)
        tensor = pad_tensor(img)
        with torch.no_grad():

            output = model(tensor)
        
        if resize:
            upsample = Resize((H, W))
        else: upsample = torch.nn.Identity()
        output = upsample(output)
        output = torch.clamp(output, 0., 1.)
        output = output[:,:, :H, :W]
        save_tensor(output, save_path+image)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./options/UHDLL.yml")

    args = parser.parse_args()
    path_options = args.config
    opt = parse(path_options)

    PATH_WEIGHT = opt['settings']['weight']
    PATH_IMAGES = opt['paths_images']['low']
    PATH_SAVE = opt['save']['path_save']

    main(weights_path=PATH_WEIGHT, path_images=PATH_IMAGES, save_path=PATH_SAVE)