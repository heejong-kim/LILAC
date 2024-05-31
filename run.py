import sys
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
import pandas as pd


def train(network, opt):
    cuda = True
    parallel = True
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    os.makedirs(f"{opt.output_fullname}/", exist_ok=True)
    if parallel:
        network = nn.DataParallel(network).to(device)

    if opt.epoch > 0:
        if len(glob.glob(f"{opt.output_fullname}/epoch{opt.epoch - 1}*.pth")) > 0:
            lastpointname = glob.glob(f"{opt.output_fullname}/epoch{opt.epoch - 1}*.pth")[0]
            network.load_state_dict(torch.load(lastpointname))
        else:
            bestepoch = np.loadtxt(os.path.join(f'' + opt.output_fullname, 'best.info'))
            bestpointname = glob.glob(f"{opt.output_fullname}/model_best.pth")[0]
            network.load_state_dict(torch.load(bestpointname))
            opt.epoch = int(bestepoch)

    optimizer = torch.optim.Adam(network.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    steps_per_epoch = opt.save_epoch_num
    writer = SummaryWriter(log_dir=f"{opt.output_fullname}")
    prev_time = time.time()
    prev_val_loss = 1e+100
    earlystoppingcount = 0

    loader_train = torch.utils.data.DataLoader(args.train_loader,
                                               batch_size=opt.batchsize, shuffle=True, num_workers=opt.num_workers,
                                               drop_last=True)

    loader_val = torch.utils.data.DataLoader(args.val_loader,
                                             batch_size=opt.batchsize, shuffle=True, num_workers=opt.num_workers,
                                             drop_last=True)

    for epoch in range(opt.epoch, opt.max_epoch):

        if earlystoppingcount > opt.earlystopping:
            break

        epoch_total_loss = []
        epoch_step_time = []

        for step, batch in enumerate(loader_train):
            step_start_time = time.time()

            if len(args.optional_meta) > 0:
                I1, I2 = batch
                input1, target1, meta1 = I1
                input2, target2, meta2 = I2
                predicted = network(input2.type(Tensor), input1.type(Tensor),
                                    meta2[:, None].type(Tensor), meta1[:, None].type(Tensor))

            else:
                I1, I2 = batch
                input1, target1 = I1
                input2, target2 = I2
                predicted = network(input2.type(Tensor), input1.type(Tensor))

            targetdiff = torch.tensor(target2 - target1)[:, None].type(Tensor)
            if opt.task_option == 'o':
                targetdiff[targetdiff > 0] = 1
                targetdiff[targetdiff == 0] = 0.5
                targetdiff[targetdiff < 0] = 0

            # Loss MSE
            optimizer.zero_grad()
            loss = args.loss(predicted, targetdiff)
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
                "\r[Epoch %d/%d] [Batch %d/%d] [ loss: %f ] ETA: %s"
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

        if ((epoch + 1) % steps_per_epoch == 0):  # (step != 0) &
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
                    if len(args.optional_meta) > 0:
                        I1, I2 = batch
                        input1, target1, meta1 = I1
                        input2, target2, meta2 = I2
                        predicted = network(torch.cat([input2.type(Tensor), input1.type(Tensor)], 1),
                                            (meta2 - meta1)[:, None].type(Tensor))

                    else:
                        I1, I2 = batch
                        input1, target1 = I1
                        input2, target2 = I2
                        predicted = network(torch.cat([input2.type(Tensor), input1.type(Tensor)], 1))

                    targetdiff = torch.tensor(target2 - target1)[:, None].type(Tensor)
                    if opt.task_option == 'o':
                        targetdiff[targetdiff > 0] = 1
                        targetdiff[targetdiff == 0] = 0.5
                        targetdiff[targetdiff < 0] = 0

                    valloss = args.loss(predicted, targetdiff)
                    valloss_total.append(valloss.item())

                log_stats([np.mean(valloss_total)], ['loss/val'], epoch, writer)
                val_loss_info = 'val loss: %.4e' % (np.mean(valloss_total))
                print(' - '.join((epoch_info, time_info, loss_info, val_loss_info)), flush=True)
                curr_val_loss = np.mean(valloss_total)
                if prev_val_loss > curr_val_loss:
                    torch.save(network.state_dict(),
                               f"{opt.output_fullname}/model_best.pth")
                    np.savetxt(f"{opt.output_fullname}/model_best.info", np.array([epoch]))
                    prev_val_loss = curr_val_loss
                    earlystoppingcount = 0  # New bottom
                else:
                    earlystoppingcount += 1
                    print(f'Early stopping count: {earlystoppingcount}')

                network.train()

    torch.save(network.state_dict(), f"{opt.output_fullname}/model_epoch{epoch}.pth")
    network.eval()


def test(network, loader, opt, overwrite=False):
    import sklearn.metrics as metrics
    from scipy.stats import pearsonr
    sigmoid = nn.Sigmoid()

    def rmse(a, b):
        return metrics.mean_squared_error(a, b, squared=False)

    savedmodelname = f"{opt.output_fullname}/model_best.pth"

    dict_metric = {'auc': metrics.roc_auc_score, 'pearson': pearsonr,
                   'rmse': rmse, 'loss': opt.loss}
    dict_task_metrics = {'o': ['loss', 'auc'],
                         't': ['loss', 'rmse'],
                         's': ['loss', 'rmse']}

    resultname = f'prediction-testset'
    run = False
    resultfilename = os.path.join(f'' + opt.output_fullname, f'{resultname}.csv')
    if os.path.exists(resultfilename):
        result = pd.read_csv(resultfilename)
        target_diff = np.array(result['target'])
        feature_diff = np.array(result['predicted'])

        for dtm in dict_task_metrics[args.task_option]:
            if dtm == 'auc' and args.task_option == 'o':
                feature_diff = sigmoid(torch.tensor(feature_diff)).numpy()

            print(f'{dtm}:', dict_metric[dtm](target_diff, feature_diff))

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

        loader_test = torch.utils.data.DataLoader(args.loader_test,
                                                  batch_size=opt.batchsize, shuffle=False, num_workers=opt.num_workers)

        tmp_stack_target = np.empty((0, 1))
        tmp_stack_predicted = np.empty((0, 1))
        tmp_stack_target1 = np.empty((0, 1))
        tmp_stack_target2 = np.empty((0, 1))

        # moved this to test subjectid key problem
        result = pd.DataFrame()
        result['subjectID'] = np.array(
            loader_test.dataset.demo[subjectid].iloc[loader_test.dataset.index_combination[:, 0]])

        for teststep, batch in enumerate(loader_test):
            sys.stdout.write(
                "\r [Batch %d/%d] "  # [ target diff: %d ]
                % (teststep,
                   len(loader_test),
                   )
            )

            if len(args.optional_meta) > 0:
                I1, I2 = batch
                input1, target1, meta1 = I1
                input2, target2, meta2 = I2
                predicted = network(torch.cat([input2.type(Tensor), input1.type(Tensor)], 1),
                                    (meta2 - meta1)[:, None].type(Tensor))

            else:
                I1, I2 = batch
                input1, target1 = I1
                input2, target2 = I2
                predicted = network(torch.cat([input2.type(Tensor), input1.type(Tensor)], 1))

            targetdiff = torch.tensor(target2 - target1)[:, None].type(Tensor)
            if opt.task_option == 'o':
                targetdiff[targetdiff > 0] = 1
                targetdiff[targetdiff == 0] = 0.5
                targetdiff[targetdiff < 0] = 0

            tmp_stack_predicted = np.append(tmp_stack_predicted,
                                            np.array((predicted).cpu().detach()),
                                            axis=0)
            tmp_stack_target = np.append(tmp_stack_target,
                                         targetdiff.cpu().detach(), axis=0)
            tmp_stack_target1 = np.append(tmp_stack_target1, np.array(target1)[:, None], axis=0)
            tmp_stack_target2 = np.append(tmp_stack_target2, np.array(target2)[:, None], axis=0)

        print('=========================')
        print(resultfilename)

        result = pd.DataFrame()
        result['target'] = tmp_stack_target
        result['predicted'] = tmp_stack_predicted
        result['target1'] = tmp_stack_target1
        result['target2'] = tmp_stack_target2
        result.to_csv(resultfilename)

        result = pd.read_csv(resultfilename)
        target_diff = np.array(result['target'])
        feature_diff = np.array(result['predicted'])

        for dtm in dict_task_metrics[args.task_option]:
            if dtm == 'auc' and args.task_option == 'o':
                feature_diff = sigmoid(torch.tensor(feature_diff)).numpy()

            print(f'{dtm}:', dict_metric[dtm](target_diff, feature_diff))


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
    parser.add_argument('--task_option', default='o', choices=['o', 't', 's'],
                        type=str, help="o: temporal ''o''rdering\n "
                                       "t: regression for ''t''ime interval\n "
                                       "s: regression with optional meta for a ''s''pecific target variable\n ")
    parser.add_argument('--targetname', default='age', type=str)
    parser.add_argument('--optional_meta', default=[], type=list,
                        help='list optional meta names to be used. csv files should include the meta data')
    parser.add_argument('--backbone_name', default='cnn_3D', type=str,
                        help="implemented models: cnn_3D, cnn_2D, resnet50_2D, resnet18_2D")

    parser.add_argument('--run_mode', default='train', choices=['train', 'eval'], help="select mode")  # required=True,

    args = parser.parse_args()

    image_size = [int(item) for item in args.image_size.split(',')]
    args.image_size = image_size

    return args


def run_setup(args):
    dict_loss = {'o': nn.BCELoss, 't': nn.MSELoss, 's': nn.MSELoss}
    dict_task = {'o': 'temporal_ordering', 't': 'regression', 's': 'regression'}

    args.loss = dict_loss[args.task_option]

    if args.optional_meta == []:
        path_pref = args.jobname + '-' + dict_task[args.task_option] + '-' + \
                    'backbone_' + args.backbone_name + '-lr' + str(args.lr) + '-seed' + str(args.seed) + '-batch' + str(
            args.batchsize)
    else:
        path_pref = args.jobname + '-' + dict_task[args.task_option] + '-' + 'meta' + '_'.join(
            args.optional_meta) + '-' + \
                    'backbone_' + args.backbone_name + '-lr' + str(args.lr) + '-seed' + str(args.seed) + '-batch' + str(
            args.batchsize)

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
        if args.run_mode == 'train':
            args.train_loader = loader2D(args, trainvaltest='train')
            args.val_loader = loader2D(args, trainvaltest='val')
        if args.run_mode == 'eval':
            args.test_loader = loader2D(args, trainvaltest='test')
    elif len(args.image_size) == 3:
        if args.run_mode == 'train':
            args.train_loader = loader3D(args, trainvaltest='train')
            args.val_loader = loader3D(args, trainvaltest='val')
        if args.run_mode == 'eval':
            args.test_loader = loader3D(args, trainvaltest='test')
    else:
        raise NotImplementedError


if __name__ == "__main__":

    args = parse_args()
    print("Hyperparameter:")
    print(args)
    run_setup(args)

    ## embryo
    args.batchsize = 64;
    args.max_epoch = 1;
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
    run_setup(args)

    # import pandas as pd
    # demo_all = pd.read_csv('/scratch/datasets/hk672/embryo/demo.csv', index_col=0)
    # demo_all['subject'] = demo_all.embryoname
    # demo_all = demo_all.drop(columns = ['embryoname','embryoidx',  'imagename-fullpath', 'resizefailed'])
    # # 'phase', 'phaseidx', target
    # demo_train = demo_all[demo_all.trainvaltest == 'train'].reset_index(drop=True).drop(columns = [ 'trainvaltest'])
    # demo_val = demo_all[demo_all.trainvaltest == 'val'].reset_index(drop=True).drop(columns = ['trainvaltest'])
    # demo_test = demo_all[demo_all.trainvaltest == 'test'].reset_index(drop=True).drop(columns = ['trainvaltest'])
    # out_dir = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release'
    # demo_train.to_csv(os.path.join(out_dir, 'demo_embryo_train.csv'))
    # demo_val.to_csv(os.path.join(out_dir, 'demo_embryo_val.csv'))
    # demo_test.to_csv(os.path.join(out_dir, 'demo_embryo_test.csv'))

    ## woundhealing
    args.batchsize = 128;
    args.max_epoch = 1;
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
    run_setup(args)

    # import pandas as pd
    # demo_all = pd.read_csv('/scratch/datasets/hk672/woundhealing/demo/demo.csv', index_col=0)
    # demo_all = demo_all.drop(columns=['subject_timepoint', 'roi', 'roifname'])
    # demo_train = demo_all[demo_all.trainvaltest == 'train'].reset_index(drop=True).drop(columns = ['trainvaltest'])
    # demo_val = demo_all[demo_all.trainvaltest == 'val'].reset_index(drop=True).drop(columns = ['trainvaltest'])
    # demo_test = demo_all[demo_all.trainvaltest == 'test'].reset_index(drop=True).drop(columns = ['trainvaltest'])
    # out_dir = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release'
    # demo_train.to_csv(os.path.join(out_dir, 'demo_woundhealing_train.csv'))
    # demo_val.to_csv(os.path.join(out_dir, 'demo_woundhealing_val.csv'))
    # demo_test.to_csv(os.path.join(out_dir, 'demo_woundhealing_test.csv'))

    ## oasis aging "oasis-aging"
    args.batchsize = 16
    args.max_epoch = 1
    args.num_workers = 8
    args.lr = 0.001
    args.backbone_name = 'cnn_3D'
    args.image_directory = '/share/sablab/nfs04/data/OASIS3/npp-preprocessed/image'
    args.task_option = 't'  # TODO loader check time interval
    args.output_directory = 'output'
    args.optional_meta = []
    args.jobname = 'oasis-aging'
    args.targetname = 'age'
    args.image_channel = 1
    args.image_size = "128, 128, 128"
    args.csv_file_train = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release/demo_oasis-aging_train.csv'
    args.csv_file_val = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release/demo_oasis-aging_val.csv'
    args.csv_file_test = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release/demo_oasis-aging_test.csv'
    run_setup(args)

    # import pandas as pd
    # demo_all_old = pd.read_csv('/share/sablab/nfs04/data/OASIS3/demo/demo-healthy-longitudinal-3D-preprocessed.csv', index_col=0)
    # demo_all = pd.DataFrame()
    # demo_all['subject'] = demo_all_old['subject-id']
    # demo_all['age'] = demo_all_old['age']
    # demo_all['fname'] = demo_all_old['fname']
    # demo_all['trainvaltest'] = demo_all_old['trainvaltest']
    #
    # demo_train = demo_all[demo_all.trainvaltest == 'train'].reset_index(drop=True).drop(columns = ['trainvaltest'])
    # demo_val = demo_all[demo_all.trainvaltest == 'val'].reset_index(drop=True).drop(columns = ['trainvaltest'])
    # demo_test = demo_all[demo_all.trainvaltest == 'test'].reset_index(drop=True).drop(columns = ['trainvaltest'])
    # out_dir = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release'
    # demo_train.to_csv(os.path.join(out_dir, 'demo_oasis-aging_train.csv'))
    # demo_val.to_csv(os.path.join(out_dir, 'demo_oasis-aging_val.csv'))
    # demo_test.to_csv(os.path.join(out_dir, 'demo_oasis-aging_test.csv'))

    ## mci w/ meta "adni-mci"
    args.batchsize = 16
    args.max_epoch = 1
    args.num_workers = 8
    args.lr = 0.001
    args.backbone_name = 'cnn_3D'
    args.image_directory = '/scratch/datasets/hk672/adni-all-3d-preprocessed/'
    args.task_option = 's'
    args.output_directory = 'output'
    args.optional_meta = ['AGExSEX']
    args.jobname = 'mci-cdrsb'
    args.targetname = 'CDRSB'
    args.image_channel = 1
    args.image_size = "128, 128, 128"
    args.csv_file_train = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release/demo_adni-mci_train.csv'
    args.csv_file_val = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release/demo_adni-mci_val.csv'
    args.csv_file_test = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release/demo_adni-mci_test.csv'
    run_setup(args)

    # import pandas as pd
    # demo_all_old = pd.read_csv('/share/sablab/nfs04/data/ADNI_mci/demo/ADNIMERGE_MCI_long_dx_conversion.csv', index_col=0)
    # demo_all = pd.DataFrame()
    # demo_all['subject'] = demo_all_old['PTID']
    # demo_all['trainvaltest'] = demo_all_old['trainvaltest']
    # demo_all['CDRSB'] = demo_all_old['CDRSB']
    # demo_all['AGExSEX'] = np.array(demo_all_old['Month']).astype('int') * np.array(demo_all_old['PTGENDER'] == 'Male').astype('int') # Month difference
    # demo_all['fname'] = np.array('I'+demo_all_old.IMAGEUID.astype('int').astype('str') + '_mni_norm.nii.gz')
    #
    # demo_train = demo_all[demo_all.trainvaltest == 'train'].reset_index(drop=True).drop(columns = ['trainvaltest'])
    # demo_val = demo_all[demo_all.trainvaltest == 'val'].reset_index(drop=True).drop(columns = ['trainvaltest'])
    # demo_test = demo_all[demo_all.trainvaltest == 'test'].reset_index(drop=True).drop(columns = ['trainvaltest'])
    # out_dir = '/home/hk672/learning-to-compare-longitudinal-images-3d/demo_for_release'
    # demo_train.to_csv(os.path.join(out_dir, 'demo_adni-mci_train.csv'))
    # demo_val.to_csv(os.path.join(out_dir, 'demo_adni-mci_val.csv'))
    # demo_test.to_csv(os.path.join(out_dir, 'demo_adni-mci_test.csv'))

    model = LILAC(args)
    print("Num of Model Parameter:", count_parameters(model))

    if args.run_mode == 'eval':
        print(' ----------------- Testing initiated -----------------')
        test(model, args)

    else:
        assert args.run_mode == 'train', "check run_mode"
        print(' ----------------- Training initiated -----------------')
        train(model, args)
