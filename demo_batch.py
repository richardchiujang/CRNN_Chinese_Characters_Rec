import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import time, glob
import cv2
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
import lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse
 
def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='lib/config/360CC_config.yaml')
    parser.add_argument('--image_path', type=str, default='images/test.jpg', help='the path to your image')
    parser.add_argument('--checkpoint', type=str, default='output/360CC/checkpoints/checkpoint_298_acc_0.9736.pth',
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def recognition(config, img, model, converter, device):

    # github issues: https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec/issues/211
    h, w = img.shape
    # fisrt step: resize the height and width of image to (32, x)
    img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)

    # second step: keep the ratio of image's text same with training
    h, w = img.shape
    w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)
    # print(preds.shape)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    print('results: {0}'.format(sim_pred))

def batch_recognition(config, imgs, model, converter, device):
    model.eval()
    # print(img.shape)
    img = torch.from_numpy(imgs)
    inp = img.view(*img.size())   # img.view(1, *img.size())
    inp = inp.to(device)
    started = time.time()
    preds = model(inp)
    finished = time.time()
    sim_preds_time = time.time()
    batch_size = inp.size(0)
    preds_size = torch.IntTensor([preds.size(0)] * batch_size)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
    for i, sim_pred in enumerate(sim_preds):
        print('results {}: {}'.format(i, sim_pred))
    print('sim_preds time: {0}'.format(time.time() - sim_preds_time))
    print('preds time: {0}'.format(finished - started))


if __name__ == '__main__':

    config, args = parse_arg()
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    model = crnn.get_crnn(config).to(device)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    # model.eval()

    img = cv2.imread(args.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    recognition(config, img, model, converter, device)

    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    stime = time.time()
    imgs_path = glob.glob('images/*.jpg')
    im_W_max = 0
    for i, img_path in enumerate(imgs_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(img.shape) # H, W
        img = cv2.resize(img, (0, 0), fx=32 / img.shape[0], fy=32 / img.shape[0], interpolation=cv2.INTER_CUBIC)
        # print(img.shape)
        im_W_max = max(im_W_max, img.shape[1])
    # print(im_W_max)

    imgs = np.zeros((len(imgs_path), 1, 32, im_W_max), dtype=np.float32)
    for i, img_path in enumerate(imgs_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (0, 0), fx=32 / img.shape[0], fy=32 / img.shape[0], interpolation=cv2.INTER_CUBIC)   # resize to 32, w
        img = cv2.copyMakeBorder(img, 0, 0, 0, im_W_max - img.shape[1], cv2.BORDER_CONSTANT, value=(255))   # padding to 32, im_W_max
        # cv2.imwrite('images/{}.jpg'.format(i), img)
        img = np.reshape(img, (32, im_W_max, 1))
        img = img.astype(np.float32)
        img = (img/255. - 0.588) / 0.193
        img = img.transpose([2, 0, 1])
        # print(img.shape)
        imgs[i] = img


    batch_recognition(config, imgs, model, converter, device)
    etime = time.time()
    print('total time: {0}'.format(etime - stime))





