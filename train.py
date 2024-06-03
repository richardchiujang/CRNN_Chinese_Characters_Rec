import os, time, datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse
import yaml
from easydict import EasyDict as edict
import lib.models.crnn as crnn
import lib.utils.utils as utils
from lib.dataset import get_dataset
from lib.core import function
import lib.config.alphabets as alphabets
# from lib.utils.utils import model_info

def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")
    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)
    # args = parser.parse_args()
    # with open(args.cfg, 'r') as f:
    with open('lib/config/360CC_config.yaml', 'r', encoding='utf-8') as f:   # 360CC_config.yaml 這裡先寫死就不用帶參數了
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config

def main():

    # load config
    config = parse_arg()
    # create output folder
    output_dict = utils.create_log_folder(config, phase='train')

    # cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # writer dict
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # construct face related neural networks
    model = crnn.get_crnn(config)
    # model_info(model)
    print('torch.cuda.device_count()', torch.cuda.device_count())
    if torch.cuda.is_available():
        device="cuda"
    else:
        device = torch.device("cpu:0")

        

    # define loss function
    criterion = torch.nn.CTCLoss()
    last_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = utils.get_optimizer(config, model)
    if isinstance(config.TRAIN.LR_STEP, list):
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        lr_scheduler = torch.optim.lr_scheduler.StepLR(    
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )

    if config.TRAIN.FINETUNE.IS_FINETUNE:
        print("finetune from", config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT)
        model_state_file = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file)   # , map_location='cpu'
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        from collections import OrderedDict
        model_dict = OrderedDict()
        for k, v in checkpoint.items():
            if 'cnn' in k:
                model_dict[k[4:]] = v
        model.cnn.load_state_dict(model_dict)
        if config.TRAIN.FINETUNE.FREEZE:
            for p in model.cnn.parameters():
                p.requires_grad = False

    elif config.TRAIN.RESUME.IS_RESUME:
        print("resume from", config.TRAIN.RESUME.FILE)
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file)  # , map_location='cpu'
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
            last_epoch = checkpoint['epoch']
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            model.load_state_dict(checkpoint)
    model = model.cuda()  # 模型完整載入後再送去 device
    # model_info(model)

    train_dataset = get_dataset(config)(config, is_train=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_dataset = get_dataset(config)(config, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )    

    best_acc = 0.1
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        now = datetime.datetime.now()
        print('epoch:', epoch, 'time:', now.strftime("%Y-%m-%d %H:%M:%S"))
        t_start_time = time.time()
        print('get_lr :', lr_scheduler.get_last_lr())
        function.train(config, train_loader, train_dataset, converter, model, criterion, optimizer, device, epoch, writer_dict, output_dict)
        lr_scheduler.step()
        print('get_last_lr :', lr_scheduler.get_last_lr(), 'train epoch time consume: ', time.time()-t_start_time)
        v_start_time = time.time()
        acc = function.validate(config, val_loader, val_dataset, converter, model, criterion, device, epoch, writer_dict, output_dict)
        print('acc validate time consume: ', time.time()-v_start_time)
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        print("is best:", is_best)
        print("best acc is:", best_acc)
        # save checkpoint
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                # "optimizer": optimizer.state_dict(),
                # "lr_scheduler": lr_scheduler.state_dict(),
                # "best_acc": best_acc,
            },  os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}_{:.9f}.pth".format(epoch, acc, lr_scheduler.get_last_lr()[0]))
        )

    writer_dict['writer'].close()

if __name__ == '__main__':

    print('strat of training time: ', time.strftime("%Y/%m/%d %H:%M:%S"))
    try:
        main()
    except KeyboardInterrupt:
        torch.cuda.empty_cache()
    print('end of training time: ', time.strftime("%Y/%m/%d %H:%M:%S"))