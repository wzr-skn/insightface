import argparse

import os
import cv2
import numpy as np
import torch
from torch.nn.functional import linear, normalize

from backbones import get_model


@torch.no_grad()
def inference(weight, name, reco_img, origin_img):
    net = get_model(name, fp16=False).to("cuda:3")
    net.load_state_dict(torch.load(weight, map_location=torch.device("cuda:3")))
    net.eval()

    assert os.path.isdir(reco_img) and os.path.isdir(origin_img)
    origin_img_list = os.listdir(origin_img)
    origin_img_path = os.path.join(origin_img, origin_img_list[0])
    ori_img = cv2.imread(origin_img_path)
    ori_img = cv2.resize(ori_img, (112, 112))
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_img = np.transpose(ori_img, (2, 0, 1))
    ori_img = torch.from_numpy(ori_img).unsqueeze(0).float()
    ori_img.div_(255).sub_(0.5).div_(0.5)
    ori_img = ori_img.to("cuda:3")
    ori_feat = net(ori_img)
    ori_norm_feat = normalize(ori_feat)

    reco_img_list = os.listdir(reco_img)
    for img_name in reco_img_list:
        img_path = os.path.join(reco_img, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        img = img.to("cuda:3")
        feat = net(img)
        norm_feat = normalize(feat)
        similarity = linear(norm_feat, ori_norm_feat)
        print("{} and {} similarity: {}".format(origin_img_path, img_path, similarity))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--reco-img', type=str, default=None)
    parser.add_argument('--origin-img', type=str, default=None)
    args = parser.parse_args()
    inference(args.weight, args.network, args.reco_img, args.origin_img)
