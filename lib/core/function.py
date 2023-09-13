from  __future__ import  absolute_import
import time
import lib.utils.utils as utils
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(config, train_loader, dataset, converter, model, criterion, optimizer, device, epoch, writer_dict=None, output_dict=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    preds_time = AverageMeter()
    criterion = criterion.to(device)
    model.train()

    end = time.time()
    for i, (inp, idx) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time() - end)
        labels = utils.get_batch_label(dataset, idx)
        inp = inp.cuda()  # to(device)
        # inference
        pres_time = time.time()
        preds = model(inp)
        preds_time.update(time.time() - pres_time)
        # compute loss
        batch_size = inp.size(0)
        # length = 一个batch中的总字符长度, text = 一个batch中的字符所对应的下标
        text, length = converter.encode(labels)                 
        preds_size = torch.IntTensor([preds.size(0)] * batch_size) # timestep * batchsize
        loss = criterion(preds, text, preds_size, length)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}] ' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s) ' \
                  'Speed {speed:.1f} samples/s ' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s) ' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f}) preds_time {preds_time.avg:.3f} '.format(
                      epoch, i+1, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, preds_time=preds_time)
            print(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()


def validate(config, val_loader, dataset, converter, model, criterion, device, epoch, writer_dict, output_dict):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # labels_time = AverageMeter()
    # inp_time = AverageMeter()
    preds_time = AverageMeter()
    conver_time_A = AverageMeter()
    # loss_time = AverageMeter()
    convert_time_B = AverageMeter()
    model.eval()

    n_correct = 0
    with torch.no_grad():
        end = time.time()
        for i, (inp, idx) in enumerate(val_loader):
            data_time.update(time.time() - end)
            # ls_time = time.time()
            labels = utils.get_batch_label(dataset, idx)
            # labels_time.update(time.time() - ls_time)
            # inps_time = time.time()
            inp = inp.cuda() # add this 這組合最快 to(device)
            # inp_time.update(time.time() - inps_time)
            # inference
            ps_time = time.time()
            preds = model(inp)  # why? .cpu() # 這裡用 cuda() 速度加倍 350/ss -> 700/sample-sec
            preds_time.update(time.time() - ps_time)
            # compute loss
            batch_size = inp.size(0)
            cons_time = time.time()
            text, length = converter.encode(labels)
            conver_time_A.update(time.time() - cons_time)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            # losss_time = time.time() 
            loss = criterion(preds, text, preds_size, length)
            losses.update(loss.item(), inp.size(0))
            # loss_time.update(time.time() - losss_time)
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            preds = preds.cpu()  # 這裡是關鍵，sim_preds_convert_time_B 1.009 (本來是8秒)
            cons_time = time.time()
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            convert_time_B.update(time.time() - cons_time)
            for pred, target in zip(sim_preds, labels):
                if pred == target:
                    n_correct += 1

            batch_time.update(time.time()-end)
            if (i + 1) % config.TEST.VALID_PRINT_FREQ == 0: # config.TEST.VALID_PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}] ' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s) ' \
                      'Speed {speed:.1f} samples/s ' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s) ' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f}) '.format(
                          epoch, i+1, len(val_loader), batch_time=batch_time,
                          speed=inp.size(0)/batch_time.val,
                          data_time=data_time, loss=losses, )
                    # labels_time=labels_time, inp_time=inp_time, 
                    # convert_time_B=convert_time_B
                    # 'sim_preds_convert_time_B {convert_time_B.avg:.3f} '
                    #'conver_time_A {conver_time_A.avg:.3f} preds_time {preds_time.avg:.3f} ' \
                    #conver_time_A=conver_time_A, preds_time=preds_time, #  loss_time=loss_time,
                    #   'labels_time {labels_time.avg:.3f} ' \ inp_time {inp_time.avg:.3f} 
                    #                         'loss_time {loss_time.avg:.3f} ' \
                print(msg)
                # print('Epoch: [{0}][{1}/{2}]'.format(epoch, i+1, len(val_loader)))

            if i == config.TEST.NUM_TEST_BATCH:
                break
            end = time.time()

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:config.TEST.NUM_TEST_DISP]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        # print('%-10s => %-10s, gt: %-10s %-10s' % (raw_pred, pred, gt, len(preds.data)))
        print('%-10s, gt: %-10s %-10s' % (pred, gt, len(preds.data)))

    num_test_sample = config.TEST.NUM_TEST_BATCH * config.TEST.BATCH_SIZE_PER_GPU
    if num_test_sample > len(dataset):
        num_test_sample = len(dataset)

    print("[#correct:{} / #total:{}]".format(n_correct, num_test_sample))
    accuracy = n_correct / float(num_test_sample)
    print('Test loss: {:.4f}, accuray: {:.4f}'.format(losses.avg, accuracy))

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_acc', accuracy, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return accuracy