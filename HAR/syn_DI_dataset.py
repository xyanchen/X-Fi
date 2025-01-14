import os
import scipy.io as scio
import glob
import cv2
import torch
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader

def decode_config(config):
    all_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14',
                    'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28',
                    'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
    all_actions = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14',
                   'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27']
    train_form = {}
    val_form = {}
    # Limitation to actions (protocol)
    if config['protocol'] == 'protocol1':  # Daily actions
        actions = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A27']
    elif config['protocol'] == 'protocol2':  # Rehabilitation actions:
        actions = ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A15', 'A16', 'A24', 'A25', 'A26']
    else:
        actions = all_actions
    # Limitation to subjects and actions (split choices)
    if config['split_to_use'] == 'random_split':
        rs = config['random_split']['random_seed']
        ratio = config['random_split']['ratio']
        for action in actions:
            np.random.seed(rs)
            idx = np.random.permutation(len(all_subjects))
            idx_train = idx[:int(np.floor(ratio*len(all_subjects)))]
            idx_val = idx[int(np.floor(ratio*len(all_subjects))):]
            subjects_train = np.array(all_subjects)[idx_train].tolist()
            subjects_val = np.array(all_subjects)[idx_val].tolist()
            for subject in all_subjects:
                if subject in subjects_train:
                    if subject in train_form:
                        train_form[subject].append(action)
                    else:
                        train_form[subject] = [action]
                if subject in subjects_val:
                    if subject in val_form:
                        val_form[subject].append(action)
                    else:
                        val_form[subject] = [action]
            rs += 1
    elif config['split_to_use'] == 'cross_scene_split':
        subjects_train = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                          'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20',
                          'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']
        subjects_val = ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
        for subject in subjects_train:
            train_form[subject] = actions
        for subject in subjects_val:
            val_form[subject] = actions
    elif config['split_to_use'] == 'cross_subject_split':
        subjects_train = config['cross_subject_split']['train_dataset']['subjects']
        subjects_val = config['cross_subject_split']['val_dataset']['subjects']
        for subject in subjects_train:
            train_form[subject] = actions
        for subject in subjects_val:
            val_form[subject] = actions
    else:
        subjects_train = config['manual_split']['train_dataset']['subjects']
        subjects_val = config['manual_split']['val_dataset']['subjects']
        actions_train = config['manual_split']['train_dataset']['actions']
        actions_val = config['manual_split']['val_dataset']['actions']
        for subject in subjects_train:
            train_form[subject] = actions_train
        for subject in subjects_val:
            val_form[subject] = actions_val

    dataset_config = {'train_dataset': {'modality': config['modality'],
                                        'split': 'training',
                                        'data_form': train_form
                                        },
                      'val_dataset': {'modality': config['modality'],
                                      'split': 'validation',
                                      'data_form': val_form}}
    return dataset_config

class MetaFi_Database:
    def __init__(self, data_root):
        self.data_root = data_root
        self.scenes = {}
        self.subjects = {}
        self.actions = {}
        self.modalities = {}
        self.load_database()

    def load_database(self):
        for scene in sorted(os.listdir(self.data_root)):
            self.scenes[scene] = {}
            for subject in sorted(os.listdir(os.path.join(self.data_root, scene))):
                self.scenes[scene][subject] = {}
                self.subjects[subject] = {}
                for action in sorted(os.listdir(os.path.join(self.data_root, scene, subject))):
                    self.scenes[scene][subject][action] = {}
                    self.subjects[subject][action] = {}
                    if action not in self.actions.keys():
                        self.actions[action] = {}
                    if scene not in self.actions[action].keys():
                        self.actions[action][scene] = {}
                    if subject not in self.actions[action][scene].keys():
                        self.actions[action][scene][subject] = {}
                    for modality in ['depth', 'rgb', 'lidar', 'mmwave', 'wifi-csi']:
                        data_path = os.path.join(self.data_root, scene, subject, action, modality)  # TODO: the path to the data file
                        self.scenes[scene][subject][action][modality] = data_path
                        self.subjects[subject][action][modality] = data_path
                        self.actions[action][scene][subject][modality] = data_path
                        if modality not in self.modalities.keys():
                            self.modalities[modality] = {}
                        if scene not in self.modalities[modality].keys():
                            self.modalities[modality][scene] = {}
                        if subject not in self.modalities[modality][scene].keys():
                            self.modalities[modality][scene][subject] = {}
                        if action not in self.modalities[modality][scene][subject].keys():
                            self.modalities[modality][scene][subject][action] = data_path


class Domain_Invariant_Dataset(Dataset):
    def __init__(self, data_base, modality, split, data_form):
        self.data_base = data_base
        # self.modality = modality.split('|')
        self.modality = modality
        print(self.modality)
        for m in self.modality:
            assert m in ['rgb','depth', 'lidar', 'mmwave', 'wifi-csi']  # 'rgb', 'infra1', 'infra2', 'depth',
        self.split = split
        self.data_source = data_form
        self.data_list = self.load_data()

    def get_scene(self, subject):
        if subject in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']:
            return 'E01'
        elif subject in ['S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']:
            return 'E02'
        elif subject in ['S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']:
            return 'E03'
        elif subject in ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']:
            return 'E04'
        else:
            raise ValueError('Subject does not exist in this dataset.')

    def load_data(self):
        data_info = tuple()
        for subject, actions in self.data_source.items():
            print(subject, actions)
            for action in actions:
                frame_list = sorted(os.listdir(os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, 'mmwave')))
                frame_num = len(frame_list)
                for idx in range(frame_num):
                    frame_idx = int(frame_list[idx].split('.')[0].split('frame')[1]) - 1
                    data_dict = {'modality': self.modality,
                                    'scene': self.get_scene(subject),
                                    'subject': subject,
                                    'action': action,
                                    'gt_path': os.path.join(self.data_base.data_root, self.get_scene(subject), subject,
                                                            action, 'ground_truth.npy'),
                                    'idx': frame_idx
                                    }
                    # print(frame_idx)
                    for mod in self.modality:
                        if mod == 'mmwave':
                            data_dict[mod+'_path'] = os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, mod, sorted(os.listdir(os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, mod)))[idx])
                        else:
                            data_dict[mod+'_path'] = os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, mod, sorted(os.listdir(os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, mod)))[frame_idx])
                    # data_dict['mmwave_filtered_path'] = os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, 'mmwave_filtered', sorted(os.listdir(os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, 'mmwave_filtered')))[idx])
                    data_info += (data_dict,)
                    # print(data_info)
                    # print(a)
        return data_info

    def read_frame(self, frame):
        _mod, _frame = os.path.split(frame)
        _, mod = os.path.split(_mod)
        if mod in ['infra1', 'infra2', 'rgb']:
            data = cv2.imread(frame)
        elif mod == 'depth':
            data = cv2.imread(frame)  # TODO: 深度和RGB的读取格式好像有区别？我忘了，待定
        elif mod == 'lidar':
            with open(frame, 'rb') as f:
                raw_data = f.read()
                data = np.frombuffer(raw_data, dtype=np.float64)
                data = data.reshape(-1, 3)
        elif mod == 'mmwave':
            with open(frame, 'rb') as f:
                raw_data = f.read()
                data = np.frombuffer(raw_data, dtype=np.float64)
                data = data.copy().reshape(-1, 5)
                # data = data[:, :3]
        elif mod == 'wifi-csi':
            data = scio.loadmat(frame)['CSIamp']
            data[np.isinf(scio.loadmat(frame)['CSIamp'])] = np.nan
            for i in range(10):  # 32
                temp_col = data[:, :, i]
                nan_num = np.count_nonzero(temp_col != temp_col)
                if nan_num != 0:
                    temp_not_nan_col = temp_col[temp_col == temp_col]
                    temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
            # csi_amp = temp_col
            # df_csi_amp = pd.DataFrame(csi_amp)
            # pd.DataFrame.fillna(methode='ffill')
            # csi_amp = df_csi_amp.values

            data = torch.tensor((data - np.min(data)) / (np.max(data) - np.min(data)))
            data = np.array(data)
        else:
            raise ValueError('Found unseen modality in this dataset.')
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        gt_numpy = np.load(item['gt_path'])
        gt_torch = torch.from_numpy(gt_numpy)
        sample = {'modality': item['modality'],
                    'scene': item['scene'],
                    'subject': item['subject'],
                    'action': item['action'],
                    'idx': item['idx'],
                    'output': gt_torch[item['idx']]
                    }
        for mod in item['modality']:
            # start_t = time.time()
            data_path = item[mod + '_path']
            if os.path.isfile(data_path):
                data_mod = self.read_frame(data_path)
                sample['input_'+mod] = data_mod
            else:
                raise ValueError('{} is not a file!'.format(data_path))
            # end_t = time.time()
            # print('Read {} takes {}s'.format(mod, end_t-start_t))
        return sample
        # sample = {'modality': ['rgb', 'depth', 'lidar', 'mmwave'],
        #           'scene': 'E01',
        #           'subject': 'S02',
        #           'action': 'A01',
        #           'idx': 6,
        #           'output': torch_tensor(17x3),
        #           'input_rgb': cv2_img(480x640x3),
        #           'input_depth': cv2_img(480x640x3),
        #           'input_lidar': numpy_array(480x640x3),
        #           'input_mmwave': numpy_array(480x640x3)
        #           }


def make_dataset(dataset_root, config):
    database = MetaFi_Database(dataset_root)
    config_dataset = decode_config(config)
    # print(config_dataset)
    train_dataset = Domain_Invariant_Dataset(database, **config_dataset['train_dataset'])
    print('/n')
    val_dataset = Domain_Invariant_Dataset(database, **config_dataset['val_dataset'])
    return train_dataset, val_dataset

# def collate_fn_padd(batch):
#     '''
#     Padds batch of variable length

#     note: it converts things ToTensor manually here since the ToTensor transform
#     assume it takes in images rather than arbitrary tensors.
#     '''
#     ## get sequence lengths
#     # for t in batch:
#         # print(t['output'].type)
#     #     print(a)
#         # print(t[0].shape,t[1].shape)
#     kpts = []
#     [kpts.append(np.array(t['output'])) for t in batch]
#     kpts = torch.FloatTensor(np.array(kpts))

#     lengths = torch.tensor([t['input_mmwave'].shape[0] for t in batch ])
#     ## padd
#     batch = [torch.Tensor(t['input_mmwave']) for t in batch ]
#     batch = torch.nn.utils.rnn.pad_sequence(batch)
#     ## compute mask
#     batch = batch.permute(1,0,2)
#     mask = (batch != 0)

#     return batch, kpts, lengths, mask

def make_dataloader(dataset, is_training, generator, batch_size, collate_fn):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=is_training,
        drop_last=is_training,
        generator=generator
    )
    return loader