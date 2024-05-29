import sys
import nibabel
from torch.utils.tensorboard import SummaryWriter
import argparse
import glob
from loader import *
from model import *
from utils import *
import torch
import numpy as np
import os
import time
import datetime
import torch.nn as nn

def train(network, loader, opt):
    cuda = True
    parallel = True
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    os.makedirs(f"{opt.save_name}/", exist_ok=True)
    if parallel:
        network = nn.DataParallel(network).to(device)

    if opt.epoch > 0 :
        if len(glob.glob(f"{opt.save_name}/epoch{opt.epoch-1}*.pth"))>0:
            lastpointname = glob.glob(f"{opt.save_name}/epoch{opt.epoch-1}*.pth")[0]
            network.load_state_dict(torch.load(lastpointname))
        else:
            bestepoch = np.loadtxt(os.path.join(f'' + opt.save_name, 'best.info'))
            bestpointname = glob.glob(f"{opt.save_name}/best.pth")[0]
            network.load_state_dict(torch.load(bestpointname))
            opt.epoch = int(bestepoch)


    optimizer = torch.optim.Adam(network.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    steps_per_epoch = opt.save_epoch_num
    writer = SummaryWriter(log_dir=f"{opt.save_name}")

    prev_time = time.time()
    prev_val_loss = 10000000
    earlystoppingcount = 0

    if not ('easy2hard' in opt.dataname):
        train_loader = loader(root=opt.image_dir, trainvaltest='train', transform=True, opt=opt)
        loader_train = torch.utils.data.DataLoader(train_loader,
              batch_size=opt.batchsize, shuffle=True, num_workers=opt.num_workers, drop_last=True)

    loader_val = torch.utils.data.DataLoader(
        loader(root=opt.image_dir, trainvaltest='val', transform=False, opt=opt),
        batch_size=opt.batchsize, shuffle=True, num_workers=opt.num_workers, drop_last=True)

    easy2hard_dict = {0: 50, 1: 50,
                      2: 40, 3: 40,
                      4: 30, 5: 30,
                      6: 20, 7: 20,
                      8: 15, 9: 15,
                      10: 10, 11: 10}

    prev_thr = -1
    for epoch in range(opt.epoch, opt.max_epoch):

        if ('easy2hard' in opt.dataname):
            if epoch < 12:
                if (prev_thr != easy2hard_dict[epoch]):
                    train_loader = loader(root=opt.image_dir, trainvaltest='train', transform=True, opt=opt,
                                          easy2hard_thr=easy2hard_dict[epoch])
                    loader_train = torch.utils.data.DataLoader(train_loader,
                                                               batch_size=opt.batchsize, shuffle=True,
                                                               num_workers=opt.num_workers, drop_last=True)
                    prev_thr = easy2hard_dict[epoch]

            if (epoch == 12) or ((prev_thr == -1) and (epoch>12)):
                train_loader = loader(root=opt.image_dir, trainvaltest='train', transform=True, opt=opt,
                                      easy2hard_thr=0)
                loader_train = torch.utils.data.DataLoader(train_loader,
                                                           batch_size=opt.batchsize, shuffle=True,
                                                           num_workers=opt.num_workers, drop_last=True)
                prev_thr = 0


        if epoch == int(opt.max_epoch / 4):
            torch.save(network.state_dict(),
                       f"{opt.save_name}/epoch{epoch}.pth")

        if earlystoppingcount > opt.earlystopping:
            break

        epoch_total_loss = []
        epoch_step_time = []

        for step, batch in enumerate(loader_train):
            step_start_time = time.time()

            if opt.deltaT:
                I1, I2 = batch
                input1, target1, age1 = I1
                input2, target2, age2 = I2
                if loader_train.dataset.positive_pairs_only:
                    assert (age1 <= age2).all(), 'age order fixed'
                deltaT = np.abs(age2 - age1)

            elif (opt.diffT or opt.diffMMSE or opt.diffCDRSB or opt.diffTSEX) and (not opt.SEX):
                I1, I2 = batch
                input1, target1, score1 = I1
                input2, target2, score2 = I2
                diffSCORE = score2 - score1

            elif (opt.diffT or opt.diffMMSE or opt.diffCDRSB) and opt.SEX:
                I1, I2 = batch
                input1, target1, score1, sex = I1
                input2, target2, score2, sex = I2
                diffSCORE = score2 - score1

            elif ~(opt.diffT or opt.diffMMSE or opt.diffCDRSB) and opt.SEX:
                I1, I2 = batch
                input1, target1, sex = I1
                input2, target2, sex = I2

            else:
                I1, I2 = batch
                input1, target1 = I1
                input2, target2 = I2

            # Feature Network
            if  'stack' in opt.backbone_name:
                if opt.deltaT:
                    predicted = network(torch.cat([input2.type(Tensor), input1.type(Tensor)], 1),
                                        deltaT[:, None].type(Tensor))
                else:
                    predicted = network(torch.cat([input2.type(Tensor), input1.type(Tensor)], 1))

            # elif 'diff' in opt.backbone_name:
            #     if opt.deltaT:
            #         predicted = network(input2.type(Tensor)-input1.type(Tensor), deltaT[:, None].type(Tensor))
            #     # TODO: diffT diffMMSE diffCDRSB naming issue....
            #
            #     else:
            #         predicted = network(input2.type(Tensor)-input1.type(Tensor))

            else: # order is fixed so age1 - age2 should be always negative
                if opt.deltaT:
                    predicted = network(input2.type(Tensor), input1.type(Tensor), deltaT[:, None].type(Tensor))

                elif (opt.diffT or opt.diffMMSE or opt.diffCDRSB or opt.diffTSEX) and (not opt.SEX):
                    predicted = network(input2.type(Tensor), input1.type(Tensor), diffSCORE[:, None].type(Tensor))

                elif (opt.diffT or opt.diffMMSE or opt.diffCDRSB) and opt.SEX:
                    predicted = network(input2.type(Tensor), input1.type(Tensor), diffSCORE[:, None].type(Tensor),
                                        sex[:, None].type(Tensor))
                elif ~(opt.diffT or opt.diffMMSE or opt.diffCDRSB) and opt.SEX:
                    predicted = network(input2.type(Tensor), input1.type(Tensor), sex[:, None].type(Tensor))

                else:
                    predicted = network(input2.type(Tensor), input1.type(Tensor))

            targetdiff = torch.tensor(target2-target1)[:, None].type(Tensor)

            # Loss MSE
            optimizer.zero_grad()
            loss = loss_mse(predicted, targetdiff)
            loss.backward()
            optimizer.step()

            epoch_total_loss.append(loss.item())

            # Log Progress
            batches_done = epoch * len(loader_train) + step
            batches_left = opt.max_epoch * len(loader_train) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [ loss: %f ] ETA: %s"  # [ target diff: %d ]
                % (
                    epoch,
                    opt.max_epoch,
                    step,
                    len(loader_train),
                    loss.item(),
                    time_left,
                )
            )
            epoch_step_time.append(time.time() - step_start_time)

        if ((epoch+1) % steps_per_epoch == 0):  # (step != 0) &
            # print epoch info
            epoch_info = '\nValidating... Step %d/%d / Epoch %d/%d' % (
                step, len(loader_train), epoch, opt.max_epoch)
            time_info = '%.4f sec/step' % np.mean(epoch_step_time)
            loss_info = 'train loss: %.4e ' % (np.mean(epoch_total_loss))

            log_stats([np.mean(epoch_total_loss)], ['loss/train'], epoch, writer)

            if not opt.no_validation:
                network.eval()
                valloss_total = []
                for valstep, batch in enumerate(loader_val):

                    if opt.deltaT:
                        I1, I2 = batch
                        input1, target1, age1 = I1
                        input2, target2, age2 = I2
                        if loader_train.dataset.positive_pairs_only:
                            assert (age1 <= age2).all(), 'age order fixed'
                        deltaT = np.abs(age2 - age1)

                    elif (opt.diffT or opt.diffMMSE or opt.diffCDRSB or opt.diffTSEX) and (not opt.SEX):
                        I1, I2 = batch
                        input1, target1, score1 = I1
                        input2, target2, score2 = I2
                        diffSCORE = score2 - score1

                    elif (opt.diffT or opt.diffMMSE or opt.diffCDRSB) and opt.SEX:
                        I1, I2 = batch
                        input1, target1, score1, sex = I1
                        input2, target2, score2, sex = I2
                        diffSCORE = score2 - score1

                    elif ~(opt.diffT or opt.diffMMSE or opt.diffCDRSB) and opt.SEX:
                        I1, I2 = batch
                        input1, target1, sex = I1
                        input2, target2, sex = I2

                    else:
                        I1, I2 = batch
                        input1, target1 = I1
                        input2, target2 = I2

                    # Feature Network
                    if 'stack' in opt.backbone_name:
                        if opt.deltaT:
                            predicted = network(torch.cat([input2.type(Tensor), input1.type(Tensor)], 1),
                                                deltaT[:, None].type(Tensor))
                        else:
                            predicted = network(torch.cat([input2.type(Tensor), input1.type(Tensor)], 1))

                    # elif 'diff' in opt.backbone_name:
                    #     if opt.deltaT:
                    #         predicted = network(input2.type(Tensor)-input1.type(Tensor), deltaT[:, None].type(Tensor))
                    #     # TODO: diffT diffMMSE diffCDRSB naming issue....
                    #
                    #     else:
                    #         predicted = network(input2.type(Tensor)-input1.type(Tensor))

                    else:  # order is fixed so age1 - age2 should be always negative
                        if opt.deltaT:
                            predicted = network(input2.type(Tensor), input1.type(Tensor), deltaT[:, None].type(Tensor))

                        elif (opt.diffT or opt.diffMMSE or opt.diffCDRSB or opt.diffTSEX) and (not opt.SEX):
                            predicted = network(input2.type(Tensor), input1.type(Tensor),
                                                diffSCORE[:, None].type(Tensor))

                        elif (opt.diffT or opt.diffMMSE or opt.diffCDRSB) and opt.SEX:
                            predicted = network(input2.type(Tensor), input1.type(Tensor),
                                                diffSCORE[:, None].type(Tensor),
                                                sex[:, None].type(Tensor))
                        elif ~(opt.diffT or opt.diffMMSE or opt.diffCDRSB) and opt.SEX:
                            predicted = network(input2.type(Tensor), input1.type(Tensor), sex[:, None].type(Tensor))

                        else:
                            predicted = network(input2.type(Tensor), input1.type(Tensor))

                    targetdiff = torch.tensor(target2 - target1)[:, None].type(Tensor)

                    valloss = loss_mse(predicted, targetdiff)
                    valloss_total.append(valloss.item())

                log_stats([np.mean(valloss_total)], ['loss/val'], epoch, writer)
                val_loss_info = 'val loss: %.4e' % (np.mean(valloss_total))
                print(' - '.join((epoch_info, time_info, loss_info, val_loss_info)), flush=True)
                network.train()
                curr_val_loss = np.mean(valloss_total)

                if prev_val_loss > curr_val_loss:
                    torch.save(network.state_dict(),
                               f"{opt.save_name}/best.pth")

                    np.savetxt(f"{opt.save_name}/best.info", np.array([epoch]))
                    prev_val_loss = curr_val_loss
                    earlystoppingcount = 0  # New bottom
                else:
                    earlystoppingcount += 1
                    print(f'Early stopping count: {earlystoppingcount}')

        if len(opt.scheduler) > 1: # TODO update this part
            scheduler.step()


    torch.save(network.state_dict(), f"{opt.save_name}/epoch{epoch}.pth")
    network.eval()

def test(network, loader, savedmodelname, opt, subjectid, overwrite=False):
    def visualize_gradcam_pair(network, opt, loader, dir_cam, cuda_idx=1, visualization=True):
        # TODO update for SEX
        fname_cam_summary = dir_cam + '-cam-summary.txt'
        fname_roi1_summary = dir_cam + '-roi1-summary.txt'
        fname_roi2_summary = dir_cam + '-roi2-summary.txt'
        # the files are matched following prediction

        # visualization = False

        import torchio as tio
        def tensor_hook(grad):
            grads['gradient']['difference'] = (grad[0].cpu().detach())
            # print(f'backward hook: {grad.size()}')

        import cv2
        savedmodelname = os.path.join(opt.save_name, 'best.pth')

        opt.batchsize = 1
        cuda = True
        parallel = True
        device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        if parallel:
            from collections import OrderedDict
            state_dict = torch.load(savedmodelname)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            network.load_state_dict(new_state_dict)
            network = network.to(device)
        else:
            network.load_state_dict(torch.load(savedmodelname))
            if cuda:
                network = network.to(device)

        network.eval()
        network.zero_grad()

        if opt.test_on_val:
            loader_test = torch.utils.data.DataLoader(
                loader(root=opt.image_dir, trainvaltest='val', transform=False, opt=opt),
                batch_size=opt.batchsize, shuffle=False, num_workers=1)

            # loader_test_roi = ADNIallconverterEasy2hardROI(root='/share/sablab/nfs04/data/ADNI_mci/',
            #                                                trainvaltest='val',
            #                                                transform=False, opt=opt)
            result_pred = pd.read_csv(os.path.join(opt.save_name, 'test-all_on_val.csv'), index_col=0)

        else:
            loader_test = torch.utils.data.DataLoader(
                loader(root=opt.image_dir, trainvaltest='test', transform=False, opt=opt),
                batch_size=opt.batchsize, shuffle=False, num_workers=1)

            # loader_test_roi = ADNIallconverterEasy2hardROI(root='/share/sablab/nfs04/data/ADNI_mci/',
            #                                                trainvaltest='test',
            #                                                transform=False, opt=opt)

            # result_pred = pd.read_csv(os.path.join(opt.save_name, 'test-all-r2.csv'), index_col=0)
            result_pred = pd.read_csv(os.path.join(opt.save_name, 'test-all.csv'), index_col=0)

        if not visualization:
            roi1_collect = []
            roi2_collect = []
            diffcam_collect = []

            for data_index in range(len(result_pred)):
                sys.stdout.write(
                    "\r [index %d/%d] "
                    % (data_index,
                       len(result_pred),
                       )
                )

                network.eval()
                network.zero_grad()
                batch = loader_test.dataset.__getitem__(data_index)
                # batchroi = loader_test_roi.__getitem__(data_index)

                # get matched ROI map

                grads = {'activation': [],
                         'gradient': {}}  # an empty dictionary

                if opt.deltaT:
                    I1, I2 = batch
                    input1, target1, age1 = I1
                    input2, target2, age2 = I2
                    deltaT = np.abs(age2 - age1)

                elif (opt.diffT or opt.diffMMSE or opt.diffCDRSB or opt.diffTSEX) and (not opt.SEX):
                    I1, I2 = batch
                    input1, target1, score1 = I1
                    input2, target2, score2 = I2
                    diffSCORE = torch.tensor([score2 - score1])

                elif (opt.diffT or opt.diffMMSE or opt.diffCDRSB) and opt.SEX:
                    I1, I2 = batch
                    input1, target1, score1, sex = I1
                    input2, target2, score2, sex = I2
                    diffSCORE = torch.tensor([score2 - score1])
                    sex = torch.tensor(np.array([sex]))
                    # print(sex)

                elif ~(opt.diffT or opt.diffMMSE or opt.diffCDRSB) and opt.SEX:
                    I1, I2 = batch
                    input1, target1, sex = I1
                    input2, target2, sex = I2
                    sex = torch.tensor(np.array([sex]))

                else:
                    I1, I2 = batch
                    input1, target1 = I1
                    input2, target2 = I2

                input1 = torch.tensor(input1)[None, :]
                input2 = torch.tensor(input2)[None, :]

                # roi1 = convert_freesurfer_fastsurfer_roi(roi1).squeeze()
                # roi2 = convert_freesurfer_fastsurfer_roi(roi2).squeeze()

                # Feature Network
                if 'featureDiff' in opt.backbone_name:

                    feature_tensor1 = -network.encoder(input1.type(Tensor).to(device))
                    feature_tensor2 = network.encoder(input2.type(Tensor).to(device))
                    feature_tensor = feature_tensor2 + feature_tensor1

                    activation_diff = feature_tensor.cpu().detach().squeeze().numpy()

                    handle_tensor = feature_tensor.register_hook(tensor_hook)
                    feature_tensor = torch.flatten(feature_tensor, 1)

                    if opt.deltaT:
                        feature_tensor = torch.concat((feature_tensor, deltaT[:, None].type(Tensor).to(device)), 1)

                    elif (opt.diffT or opt.diffMMSE or opt.diffCDRSB or opt.diffTSEX) and (not opt.SEX):
                        feature_tensor = torch.concat((feature_tensor, diffSCORE[:, None].type(Tensor).to(device)), 1)

                    elif (opt.diffT or opt.diffMMSE or opt.diffCDRSB) and opt.SEX:
                        feature_tensor = torch.concat((feature_tensor, diffSCORE[:, None].type(Tensor),
                                                       sex[:, None].type(Tensor).to(device)), 1)
                    elif ~(opt.diffT or opt.diffMMSE or opt.diffCDRSB) and opt.SEX:
                        feature_tensor = torch.concat((feature_tensor, sex[:, None].type(Tensor).to(device)), 1)

                    else:
                        assert False, "not implemented"

                    # predicted = sigmoid(network.linear(feature_tensor))
                    predicted = network.linear(feature_tensor)

                else:
                    print("WARNING: Not implemented yet")
                    assert False, "CHECK MODELNAME"

                predicted.backward()
                handle_tensor.remove()

                gradient_diff = grads['gradient']['difference'].squeeze().numpy()

                # save elementwise gradcam:
                gradcam_activation_elementwise = (activation_diff * gradient_diff).sum(0)
                # roi_avg = calculate_roi_values(roi1, roi2, gradcam_activation_elementwise)
                # roi1_collect.append(roi_avg[:, 0])
                # roi2_collect.append(roi_avg[:, 1])
                diffcam_collect.append(gradcam_activation_elementwise.flatten())

            # np.savetxt(fname_roi1_summary, np.array(roi1_collect))
            # np.savetxt(fname_roi2_summary, np.array(roi2_collect))
            np.savetxt(fname_cam_summary, np.array(diffcam_collect))

    from sklearn.metrics import roc_auc_score as auc

    resultname = f'test-all'
    run = False
    resultfilename = os.path.join(f'' + opt.save_name, f'{resultname}.csv')
    if os.path.exists(resultfilename):
        result = pd.read_csv(resultfilename)
        stack_target_diff = np.array(result['gt-target'])
        stack_feature_diff = np.array(result['predicted'])
        print(f"MSE: {loss_mse(torch.tensor(stack_target_diff), torch.tensor(stack_feature_diff))}")
        print(f'AUC: {auc(result["gt-target"] > 0, result["predicted"]>0):.3f}')


    if not os.path.exists(resultfilename) or overwrite:
        run = True

    if run:
        print('working on ', resultname)
        cuda = True
        parallel = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        if parallel:
            network = nn.DataParallel(network).to(device)
            network.load_state_dict(torch.load(savedmodelname))
        else:
            network.load_state_dict(torch.load(savedmodelname))
            if cuda:
                network = network.cuda()

        network.eval()

        loader_test = torch.utils.data.DataLoader(
            loader(root=opt.image_dir, trainvaltest='test', transform=False, opt=opt),
            batch_size=opt.batchsize, shuffle=False, num_workers=opt.num_workers)

        tmp_stack_target_diff = np.empty((0, 1))
        tmp_stack_target1 = np.empty((0, 1))
        tmp_stack_target2 = np.empty((0, 1))

        if 'Multitask' in opt.backbone_name:
            tmp_stack_feature_diff = np.empty((0, 3))
        else:
            tmp_stack_feature_diff = np.empty((0, 1))

        # moved this to test subjectid key problem
        result = pd.DataFrame()
        result['subjectID'] = np.array(loader_test.dataset.demo[subjectid].iloc[loader_test.dataset.index_combination[:, 0]])

        for teststep, batch in enumerate(loader_test):
            sys.stdout.write(
                "\r [Batch %d/%d] "  # [ target diff: %d ]
                % (teststep,
                   len(loader_test),
                   )
            )

            if opt.deltaT:
                I1, I2 = batch
                input1, target1, age1 = I1
                input2, target2, age2 = I2
                if loader_train.dataset.positive_pairs_only:
                    assert (age1 <= age2).all(), 'age order fixed'
                deltaT = np.abs(age2 - age1)

            elif (opt.diffT or opt.diffMMSE or opt.diffCDRSB or opt.diffTSEX) and (not opt.SEX):
                I1, I2 = batch
                input1, target1, score1 = I1
                input2, target2, score2 = I2
                diffSCORE = score2 - score1

            elif (opt.diffT or opt.diffMMSE or opt.diffCDRSB) and opt.SEX:
                I1, I2 = batch
                input1, target1, score1, sex = I1
                input2, target2, score2, sex = I2
                diffSCORE = score2 - score1

            elif ~(opt.diffT or opt.diffMMSE or opt.diffCDRSB) and opt.SEX:
                I1, I2 = batch
                input1, target1, sex = I1
                input2, target2, sex = I2

            else:
                I1, I2 = batch
                input1, target1 = I1
                input2, target2 = I2

            # Feature Network
            if  'stack' in opt.backbone_name:
                if opt.deltaT:
                    predicted = network(torch.cat([input2.type(Tensor), input1.type(Tensor)], 1),
                                        deltaT[:, None].type(Tensor))
                else:
                    predicted = network(torch.cat([input2.type(Tensor), input1.type(Tensor)], 1))

            # elif 'diff' in opt.backbone_name:
            #     if opt.deltaT:
            #         predicted = network(input2.type(Tensor)-input1.type(Tensor), deltaT[:, None].type(Tensor))
            #     # TODO: diffT diffMMSE diffCDRSB naming issue....
            #
            #     else:
            #         predicted = network(input2.type(Tensor)-input1.type(Tensor))

            else: # order is fixed so age1 - age2 should be always negative
                if opt.deltaT:
                    predicted = network(input2.type(Tensor), input1.type(Tensor), deltaT[:, None].type(Tensor))

                elif (opt.diffT or opt.diffMMSE or opt.diffCDRSB or opt.diffTSEX) and (not opt.SEX):
                    predicted = network(input2.type(Tensor), input1.type(Tensor), diffSCORE[:, None].type(Tensor))

                elif (opt.diffT or opt.diffMMSE or opt.diffCDRSB) and opt.SEX:
                    predicted = network(input2.type(Tensor), input1.type(Tensor), diffSCORE[:, None].type(Tensor),
                                        sex[:, None].type(Tensor))
                elif ~(opt.diffT or opt.diffMMSE or opt.diffCDRSB) and opt.SEX:
                    predicted = network(input2.type(Tensor), input1.type(Tensor), sex[:, None].type(Tensor))

                else:
                    predicted = network(input2.type(Tensor), input1.type(Tensor))


            targetdiff = torch.tensor(target2-target1)[:, None].type(Tensor)


            tmp_stack_feature_diff = np.append(tmp_stack_feature_diff,
                                               np.array((predicted).cpu().detach()),
                                               axis=0)

            tmp_stack_target_diff = np.append(tmp_stack_target_diff,
                                              targetdiff.cpu().detach(), axis=0)
            tmp_stack_target1 = np.append(tmp_stack_target1, np.array(target1)[:, None], axis=0)
            tmp_stack_target2 = np.append(tmp_stack_target2, np.array(target2)[:, None], axis=0)

        print('=========================')
        print(resultfilename)

        result['gt-target'] = tmp_stack_target_diff.squeeze()
        result['target1'] = tmp_stack_target1.squeeze()
        result['target2'] = tmp_stack_target2.squeeze()
        result['predicted'] = tmp_stack_feature_diff.squeeze()
        print(f"MSE: {loss_mse(torch.tensor(tmp_stack_target_diff), torch.tensor(tmp_stack_feature_diff))}")
        print(f'AUC: {auc(result["gt-target"] > 0, result["predicted"]>0):.3f}')

        result.to_csv(resultfilename)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobname', default='lilac', type=str, help="name of job")  # , required=True)

    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    parser.add_argument('--earlystopping', default=10, type=int, help="early stopping criterion")
    parser.add_argument('--batchsize', default=16, type=int)
    parser.add_argument('--max_epoch', default=300, type=int, help="max epoch")
    parser.add_argument('--epoch', default=0, type=int, help="starting epoch")
    parser.add_argument('--save_epoch_num', default=1, type=int, help="validate and save every N epoch")

    parser.add_argument('--image_directory', default='./datasets', type=str)  # , required=True)
    parser.add_argument('--csv_file_train', default='./datasets/demo_oasis_train.csv', type=str,
                        help="csv file for training set")  # , required=True)
    parser.add_argument('--csv_file_val', default='./datasets/demo_oasis_val.csv', type=str,
                        help="csv file for validation set")  # , required=True)
    parser.add_argument('--csv_file_test', default='./datasets/demo_oasis_test.csv', type=str,
                        help="csv file for testing set")  # , required=True)
    parser.add_argument('--output_directory', default='./output', type=str,
                        help="directory path for saving model and outputs")  # , required=True)

    parser.add_argument('--image_size', default="128, 128, 128", type=str, help="w,h for 2D and w,h,d for 3D")
    parser.add_argument('--image_channel', default=1, type=int)
    parser.add_argument('--task_option', default='o', choices=['o', 't', 'm'],
                                                                     type=str, help="o: temporal ordering\n "
                                                                     "t: regression for time interval\n "
                                                                     "m: regression with optional meta for a specific target variable\n ")
    parser.add_argument('--targetname', default='age', type=str)
    parser.add_argument('--optional_meta', default=[], type=list, help='list optional meta names to be used. csv files should include the meta data')
    parser.add_argument('--backbone_name', default='cnn_3D', type=str,
                        help="implemented models: cnn_3D, cnn_2D, resnet50_2D, resnet18_2D")

    parser.add_argument('--run_mode', default='train', choices=['train', 'eval'], help="select mode") #  required=True,

    args = parser.parse_args()

    image_size = [int(item) for item in args.image_size.split(',')]
    args.image_size = image_size

    return args

def run_setup(args):
    dict_loss= {'o': nn.BCELoss, 't':nn.MSELoss, 'm':nn.MSELoss}
    dict_task = {'o': 'temporal_ordering', 't':'regression', 'm':'regression'}

    args.loss = dict_loss[args.task_option]

    if args.optional_meta == []:
        path_pref = args.jobname + '-' + dict_task[args.task_option] + '-' + \
                    'backbone_' + args.backbone_name + '-lr' + str(args.lr) + '-seed' + str(args.seed) + '-batch' + str(args.batchsize)
    else:
        path_pref = args.jobname + '-' + dict_task[args.task_option] + '-' + 'meta' + '_'.join(args.optional_meta) + \
                    'backbone_' + args.backbone_name + '-lr' + str(args.lr) + '-seed' + str(args.seed) + '-batch' + str(args.batchsize)

    args.output_fullname = os.path.join(args.output_directory, path_pref)
    os.makedirs(args.output_fullname, exist_ok=True)

    # check path
    assert os.path.exists(args.image_directory), "incorrect image directory path"

    # set up seed
    set_manual_seed(args.seed)

    # set up GPU
    print(f"Available GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
        print("!! NO GPU AVAILABLE !!")

    # string to list
    image_size = [int(item) for item in args.image_size.split(',')]
    args.image_size = image_size

    if len(args.image_size) == 2:
        args.loader = loader2D(args)
    elif len(args.image_size) == 3:
        args.loader = loader3D(args)
    else:
        raise NotImplementedError

if __name__ == "__main__":

    args = parse_args()
    print("Hyperparameter:")
    print(args)
    run_setup(args)

    ## embryo
    args.batchsize = 64;
    args.max_epoch = 40;
    args.num_workers = 8;
    args.targetname = 'phaseidx';
    args.optional_meta = []
    args.lr = 0.001;
    args.backbone_name = 'cnn_2D';
    args.save_epoch_num = 1;
    args.task_option = 'o'
    args.output_directory = 'output'
    args.jobname = 'embryo'
    args.image_directory = '/scratch/datasets/hk672/embryo';
    args.image_size = "224, 224"
    args.csv_file_train = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release/demo_embryo_train.csv'
    args.csv_file_val = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release/demo_embryo_val.csv'
    args.csv_file_test = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release/demo_embryo_test.csv'

    import pandas as pd
    demo_all = pd.read_csv('/scratch/datasets/hk672/embryo/demo.csv', index_col=0)
    demo_all['subject'] = demo_all.embryoname
    demo_all = demo_all.drop(columns = ['embryoname','embryoidx',  'imagename-fullpath', 'resizefailed'])
    # 'phase', 'phaseidx', target
    demo_train = demo_all[demo_all.trainvaltest == 'train'].reset_index(drop=True).drop(columns = [ 'trainvaltest'])
    demo_val = demo_all[demo_all.trainvaltest == 'val'].reset_index(drop=True).drop(columns = ['trainvaltest'])
    demo_test = demo_all[demo_all.trainvaltest == 'test'].reset_index(drop=True).drop(columns = ['trainvaltest'])
    out_dir = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release'
    demo_train.to_csv(os.path.join(out_dir, 'demo_embryo_train.csv'))
    demo_val.to_csv(os.path.join(out_dir, 'demo_embryo_val.csv'))
    demo_test.to_csv(os.path.join(out_dir, 'demo_embryo_test.csv'))

    ## woundhealing
    args.batchsize = 128;
    args.max_epoch = 40;
    args.num_workers = 8;
    args.targetname = 'timepoint';
    args.optional_meta = []
    args.lr = 0.001;
    args.backbone_name = 'cnn_2D';
    args.save_epoch_num = 1;
    args.task_option = 'o'
    args.output_directory = 'output'
    args.jobname = 'woundhealing'
    args.image_directory = '/scratch/datasets/hk672/woundhealing/data_preprocessed';
    args.image_size = "224, 224"
    args.csv_file_train = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release/demo_woundhealing_train.csv'
    args.csv_file_val = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release/demo_woundhealing_val.csv'
    args.csv_file_test = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release/demo_woundhealing_test.csv'

    import pandas as pd
    demo_all = pd.read_csv('/scratch/datasets/hk672/woundhealing/demo/demo.csv', index_col=0)
    demo_all = demo_all.drop(columns=['subject_timepoint', 'roi', 'roifname'])
    demo_train = demo_all[demo_all.trainvaltest == 'train'].reset_index(drop=True).drop(columns = ['trainvaltest'])
    demo_val = demo_all[demo_all.trainvaltest == 'val'].reset_index(drop=True).drop(columns = ['trainvaltest'])
    demo_test = demo_all[demo_all.trainvaltest == 'test'].reset_index(drop=True).drop(columns = ['trainvaltest'])
    out_dir = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release'
    demo_train.to_csv(os.path.join(out_dir, 'demo_woundhealing_train.csv'))
    demo_val.to_csv(os.path.join(out_dir, 'demo_woundhealing_val.csv'))
    demo_test.to_csv(os.path.join(out_dir, 'demo_woundhealing_test.csv'))

    ## oasis aging "oasis-aging"
    args.batchsize = 16
    args.max_epoch = 1
    args.num_workers = 8
    args.lr = 0.001
    args.backbone_name = 'cnn_3D'
    args.image_directory = '/share/sablab/nfs04/data/OASIS3/npp-preprocessed/image'
    args.task_option = 't' # TODO loader check time interval
    args.output_directory = 'output'
    args.optional_meta = []
    args.jobname = 'oasis-aging'
    args.targetname = 'age'
    args.image_channel = 1
    args.image_size = "128, 128, 128"
    args.csv_file_train = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release/demo_oasis-aging_train.csv'
    args.csv_file_val = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release/demo_oasis-aging_val.csv'
    args.csv_file_test = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release/demo_oasis-aging_test.csv'

    import pandas as pd
    demo_all_old = pd.read_csv('/share/sablab/nfs04/data/OASIS3/demo/demo-healthy-longitudinal-3D-preprocessed.csv', index_col=0)
    demo_all = pd.DataFrame()
    demo_all['subject'] = demo_all_old['subject-id']
    demo_all['age'] = demo_all_old['age']
    demo_all['fname'] = demo_all_old['fname']
    demo_all['trainvaltest'] = demo_all_old['trainvaltest']

    demo_train = demo_all[demo_all.trainvaltest == 'train'].reset_index(drop=True).drop(columns = ['trainvaltest'])
    demo_val = demo_all[demo_all.trainvaltest == 'val'].reset_index(drop=True).drop(columns = ['trainvaltest'])
    demo_test = demo_all[demo_all.trainvaltest == 'test'].reset_index(drop=True).drop(columns = ['trainvaltest'])
    out_dir = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release'
    demo_train.to_csv(os.path.join(out_dir, 'demo_oasis-aging_train.csv'))
    demo_val.to_csv(os.path.join(out_dir, 'demo_oasis-aging_val.csv'))
    demo_test.to_csv(os.path.join(out_dir, 'demo_oasis-aging_test.csv'))


    ## mci w/ meta "adni-mci"
    model = LILAC(args)
    print("Num of Model Parameter:", count_parameters(model))
    loader = args.loader


    if args.run_mode == 'eval':
        print(' -----------------Testing initiated -----------------')

        # TODO update test
        test(network, loader, savedmodelname, opt, subjectid=dict_subjectname[opt.dataname])
        # test(network, loader, savedmodelname, opt, subjectid, overwrite=False):

        if opt.visualize_gpu != -1:
            dir_cam = os.path.join(results_path, f'CAM-comparison',
                                   f'{opt.dataname}-{opt.targetname}-{opt.pooling}-{opt.backbone_name}')

            os.makedirs(dir_cam, exist_ok=True)
            visualize_gradcam_pair(network, opt, loader, dir_cam, cuda_idx=opt.visualize_gpu, visualization=False)

    else:
        assert args.run_mode == 'train', "check run_mode"
        print(' -----------------Training initiated -----------------')
        train(network, dict_dataloader[opt.dataname], opt)

