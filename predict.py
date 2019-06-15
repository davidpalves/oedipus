import torch
from PIL import Image
from train import Net
import numpy as np
from utils import DEVICE, CHARS
from torchvision.transforms import functional
import argparse


class Predictor(object):
    def __init__(self, model_path, gpu=True):
        self.net = Net(gpu)
        self.net.load(model_path)

    def identify(self, im_path):
        im = Image.open(im_path)

        # to tensor
        np_img = np.asarray(im)
        image = np_img.transpose((2, 0, 1))  # H x W x C  -->  C x H x W
        im = torch.from_numpy(image).float()

        # normalize
        im = functional.normalize(im, [127.5, 127.5, 127.5], [128, 128, 128])
        if self.net.device != 'cpu':
            im = im.to(DEVICE)

        with torch.no_grad():
            xb = im.unsqueeze(0)
            out = self.net(xb).squeeze(0).view(4, 19)
            _, predicted = torch.max(out, 1)
            ans = [CHARS[i] for i in predicted.tolist()]
            return ans


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict captchas using CNN')

    parser.add_argument('--path', '-p', help='Path to the captcha image',
                        type=str, required=True)
    parser.add_argument('--debug', help='Run in debug mode',
                        action='store_true')
    args = parser.parse_args()

    man = Predictor('pretrained')
    if args.debug:
        captcha = Image.open(args.path)
        captcha.show()
        print(man.identify(args.path))
    else:
        print("Not a valid path!")
