import os
import os.path
import sys
import argparse
import time
import yaml
import numpy as np
import torch
from torch import optim

from mmfi_lib.mmfi import make_dataset, make_dataloader
from mmfi_lib.evaluate import calulate_error
from model import TemporalModel
import loss
from generators import ChunkedGenerator
from skeleton import Skeleton


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - torch.from_numpy(np.array([1, h / w]))


# def predict_3d_joints(npz_path):
#     # t_start = time.time()
#     pose = np.load(npz_path, allow_pickle=True)
#     kps_2d_pxl = pose.get('keypoints_2d')
#     kps_2d = normalize_screen_coordinates(kps_2d_pxl, w=640, h=480)
#     kps_3d = fair_infer_3d_kps(kps_2d)[0]
#     # t_use = time.time() - t_start
#     # print('Using time {:.03f}'.format(t_use))
#     return kps_3d


def calculate_errors(data_dir):
    mpjpe_all = []
    pa_mpjpe_all = []
    for sample in sorted(os.listdir(data_dir)):
        sample_path = os.path.join(data_dir, sample)
        sample_data = np.load(sample_path, allow_pickle=True)
        predict = sample_data.get('joints')
        gt = sample_data.get('gt')
        mpjpe, pa_mpjpe = calulate_error(predict, gt)
        mpjpe_all.append(mpjpe)
        pa_mpjpe_all.append(pa_mpjpe)
    mpjpe_all = np.array(mpjpe_all)
    pa_mpjpe_all = np.array(pa_mpjpe_all)
    mpjpe_mean = np.mean(mpjpe_all)
    pa_mpjpe_mean = np.mean(pa_mpjpe_all)
    # print('MPJPE: {}, PA-MPJPE: {}'.format(mpjpe_mean, pa_mpjpe_mean))
    return mpjpe_mean, pa_mpjpe_mean


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Code Implementation for VideoPose3D on MMFi-RGB Dataset")
    # parser.add_argument("protocol_setting", type=str, help="Protocol and setting for this trial")
    # args = parser.parse_args()

    round_list = ["round3"]
    # round_idx = "round1"

    setting_list = ["p2s3", "p3s1", "p3s2", "p3s3"]
    # setting_list = ["p1s1", "p1s2", "p1s3", "p2s1", "p2s2", "p2s3", "p3s1", "p3s2", "p3s3"]

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    dataset_root = "/mnt/mmfi_dataset"

    checkpoint_frequency = 10

    for protocol_setting in setting_list:
        for round_idx in round_list:
            checkpoint_dir = os.path.join("/data2/code/mmfi_rgb_save", protocol_setting, round_idx)
            os.makedirs(checkpoint_dir, exist_ok=True)

            # if not os.path.exists(os.path.join(checkpoint_dir, "epoch_60.bin")):

            # protocol_setting = "p1s3"

            # protocol_setting = args.protocol_setting

            '''
            Train a VideoPose3D model from scratch
            '''
            with open("configs/" + protocol_setting + ".yaml", 'r') as fd:
                config = yaml.load(fd, Loader=yaml.FullLoader)

            train_dataset, val_dataset = make_dataset(dataset_root, config)

            # rng_generator = torch.manual_seed(config['init_rand_seed'])
            rng_generator = torch.manual_seed(127)
            train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
            val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['validation_loader'])

            kps_left = [1, 3, 5, 7, 9, 11, 13, 15]
            kps_right = [2, 4, 6, 8, 10, 12, 14, 16]

            joints_left = [4, 5, 6, 11, 12, 13]
            joints_right = [1, 2, 3, 14, 15, 16]

            h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                              16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                                     joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                                     joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
            h36m_joints_to_remove = [4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31]

            lr = 0.001
            lr_decay = 0.95

            model_architecture = [3,3,3,3]
            channels = 1024

            num_joints_in = 17
            num_features_in = 2
            num_joints_out = 17

            model_pos_train = TemporalModel(num_joints_in, num_features_in, num_joints_out, filter_widths=model_architecture,
                                            causal=False, dropout=0.25, channels=channels, dense=False)
            model_pos = TemporalModel(num_joints_in, num_features_in, num_joints_out, filter_widths=model_architecture,
                                            causal=False, dropout=0.25, channels=channels, dense=False)

            receptive_field = model_pos.receptive_field()
            print('INFO: Receptive field: {} frames'.format(receptive_field))
            pad = (receptive_field - 1) // 2  # Padding on each side
            causal_shift = 0  # causal=False

            model_params = 0
            for parameter in model_pos.parameters():
                model_params += parameter.numel()
            print('INFO: Trainable parameter count:', model_params)

            if torch.cuda.is_available():
                model_pos = model_pos.cuda()
                model_pos_train = model_pos_train.cuda()

            optimizer = optim.Adam(model_pos_train.parameters(), lr=lr, amsgrad=True)

            losses_3d_train = []
            losses_3d_train_eval = []
            losses_3d_valid = []

            epoch = 0
            initial_momentum = 0.1
            final_momentum = 0.001

            num_epochs = 60

            export_training_curves = True

            poses_3d_in = []
            poses_2d_in = []

            if os.path.isfile(os.path.join("./data_cache", protocol_setting, "train", "poses_2d_in.pt")) and \
                    os.path.isfile(os.path.join("./data_cache", protocol_setting, "train", "poses_3d_in.pt")):
                poses_2d_in = torch.load(os.path.join("./data_cache", protocol_setting, "train", "poses_2d_in.pt"))
                poses_3d_in = torch.load(os.path.join("./data_cache", protocol_setting, "train", "poses_3d_in.pt"))
            else:
                for idx, data in enumerate(train_loader):
                    poses_2d_in = normalize_screen_coordinates(data['input_rgb'], w=640, h=480)
                    poses_3d_in = data['output']

                os.makedirs(os.path.join("./data_cache", protocol_setting, "train"), exist_ok=True)
                torch.save(poses_2d_in, os.path.join("./data_cache", protocol_setting, "train", "poses_2d_in.pt"))
                torch.save(poses_3d_in, os.path.join("./data_cache", protocol_setting, "train", "poses_3d_in.pt"))
                print("Saved the data cache into " + protocol_setting)

            train_generator = ChunkedGenerator(1024, None, poses_3d_in, poses_2d_in, 1,
                                               pad=pad, causal_shift=causal_shift, shuffle=True, random_seed=5678, augment=True,
                                               kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                               joints_right=joints_right)

            while epoch < num_epochs:
                start_time = time.time()
                epoch_loss_3d_train = 0
                epoch_loss_traj_train = 0
                epoch_loss_2d_train_unlabeled = 0
                N = 0
                model_pos_train.train()

                for _, batch_3d, batch_2d in train_generator.next_epoch():
                    inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                    # inputs_3d[:, :, 0] = 0

                    optimizer.zero_grad()

                    # Predict 3D poses
                    predicted_3d_pos = model_pos_train(inputs_2d)
                    loss_3d_pos = loss.mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    loss_total = loss_3d_pos
                    loss_total.backward()

                    optimizer.step()

                losses_3d_train.append(epoch_loss_3d_train / N)

                elapsed = (time.time() - start_time) / 60

                print('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000))

                # Decay learning rate exponentially
                lr *= lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_decay
                epoch += 1

                # Decay BatchNorm momentum
                momentum = initial_momentum * np.exp(-epoch / num_epochs * np.log(initial_momentum / final_momentum))
                model_pos_train.set_bn_momentum(momentum)

                # Save checkpoint if necessary
                if epoch % checkpoint_frequency == 0:
                    chk_path = os.path.join(checkpoint_dir, 'epoch_{}.bin'.format(epoch))
                    print('Saving checkpoint to', chk_path)

                    torch.save({
                        'epoch': epoch,
                        'lr': lr,
                        'model_pos_train': model_pos_train.state_dict(),
                    }, chk_path)

                # Save training curves after every epoch, as .png images (if requested)
                if export_training_curves and epoch > 3:
                    # if 'matplotlib' not in sys.modules:
                    import matplotlib

                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt

                    plt.figure()
                    epoch_x = np.arange(3, len(losses_3d_train)) + 1
                    plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
                    # plt.plot(epoch_x, losses_3d_train_eval[3:], color='C0')
                    # plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
                    # plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
                    plt.ylabel('MPJPE (m)')
                    plt.xlabel('Epoch')
                    plt.xlim((3, epoch))
                    plt.savefig(os.path.join(checkpoint_dir, 'loss_3d.png'))

                    plt.close('all')

    '''
    Generate the predicted 3D joints using a trained VideoPose3D model
    '''
    # for cfg_file in os.listdir('./configs'):
    #     with open(os.path.join('./configs', cfg_file), 'r') as fd:
    #         config = yaml.load(fd, Loader=yaml.FullLoader)
    #
    #     train_dataset, val_dataset = make_dataset(dataset_root, config)
    #
    #     rng_generator = torch.manual_seed(config['init_rand_seed'])
    #     # train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
    #     val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['validation_loader'])
    #
    #     for idx, sample in enumerate(val_loader):
    #         # sample = val_dataset[idx]
    #         keypoints_path = os.path.join(dataset_root, sample['scene'][0], sample['subject'][0], sample['action'][0], 'Keypoints', 'rgb.npz')
    #         joints = predict_3d_joints(keypoints_path)
    #         save_dir = os.path.join(save_root, config['protocol'], config['split_to_use'], config['modality'])
    #         os.makedirs(save_dir, exist_ok=True)
    #         sample_name = sample['subject'][0] + '_' + sample['action'][0] + '.npz'
    #         np.savez(os.path.join(save_dir, sample_name), joints=joints, gt=sample['output'][0])
    #         print('result of {} has been saved'.format(sample_name))

    # def get_scene(subject):
    #     if subject in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']:
    #         return 'E01'
    #     elif subject in ['S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']:
    #         return 'E02'
    #     elif subject in ['S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']:
    #         return 'E03'
    #     elif subject in ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']:
    #         return 'E04'
    #     else:
    #         raise ValueError('Subject does not exist in this dataset.')

    '''
    Compare the predicted 3D joints with the ground truth pose.
    '''
    # for _protocol in sorted(os.listdir(save_root)):
    #     protocol = os.path.join(save_root, _protocol)
    #     for _split in sorted(os.listdir(protocol)):
    #         split = os.path.join(protocol, _split)
    #         rgb_dir = os.path.join(split, 'rgb')
    #         mpjpe, pa_mpjpe = calculate_errors(rgb_dir)
    #         '''
    #         Save the errors
    #         '''
    #         ## Save the evaluation results
    #         # with open("results_record_3.txt", "a") as f:
    #         #     f.write("{pro}  {spl} : \n".format(pro=_protocol, spl=_split))
    #         #     f.write("{res_1}  {res_2} \n".format(res_1=mpjpe, res_2=pa_mpjpe))




