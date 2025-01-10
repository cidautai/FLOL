import os
import lpips
from glob import glob
import argparse
import numpy as np

class Metric:
    def __init__(self, path_gt=None, path_pred=None, extension=None):
        self.list_gt=sorted(glob(os.path.join(path_gt, f'*{extension}')))
        self.list_pred=sorted(glob(os.path.join(path_pred, f'*{extension}')))
    
    def calculate_metric(self):
        eval_lpips=[]
        lpips_model = lpips.LPIPS(net='alex').to("cuda")
        for gt, pred in zip(self.list_gt, self.list_pred):
            im1=lpips.im2tensor(lpips.load_image(gt)).cuda()
            im2=lpips.im2tensor(lpips.load_image(pred)).cuda()
            lpips_model = lpips.LPIPS(net='alex').to("cuda")
            lpips_distance = lpips_model(im1, im2)
            eval_lpips.append(lpips_distance.item())
        print(np.mean(eval_lpips))
        return np.mean(eval_lpips)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="LPIPS script")
    parser.add_argument('-g', '--path-gt', type=str, default='/mnt/valab-datasets/Low_Light_Enhancement_Datasets/UHD-LL/UHD-LL_down4/testing_set/gt', 
                        help="Path to your gt images (e.g., desktop/train).")
    parser.add_argument('-p', '--path-pred', type=str, default='/home/juaben/Training/Experiments/results/paired/UHD-LL', 
                        help="Path to your predicted images (e.g., desktop/train).")
    
    parser.add_argument('-e', '--extension', type=str, default='.jpg',
                        help="Extension of your images (e.g., '.jpg', '.png').")

    args=parser.parse_args()

    metric=Metric(path_gt=args.path_gt, path_pred=args.path_pred, extension=args.extension)

    metric.calculate_metric()