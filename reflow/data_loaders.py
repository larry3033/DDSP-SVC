import os
import random
import re
import numpy as np
import librosa
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset
import concurrent.futures


def get_npy_shape(file_path):
    with open(file_path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape = np.lib.format.read_array_header_1_0(f)[0]
        elif version == (2, 0):
            shape = np.lib.format.read_array_header_2_0(f)[0]
        else:
            raise ValueError("Unsupported .npy file version")
    return shape
        
def traverse_dir(
        root_dir,
        extensions,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


def get_data_loaders(args, whole_audio=False):
    use_spk = bool(args.model.get('use_spk', False))
    use_spk_tokens = bool(args.model.get('use_spk_tokens', False)) if use_spk else False
    data_train = AudioDataset(
        args.data.train_path,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=whole_audio,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
        device=args.train.cache_device,
        fp16=args.train.cache_fp16,
        use_aug=True,
        use_spk=use_spk,
        use_spk_tokens=use_spk_tokens,
        spk_cross_prob=args.train.get('spk_cross_prob', 0.5),
        spk_noise=args.train.get('spk_noise', 0.02),
        spk_drop_prob=args.train.get('spk_drop_prob', 0.1))
    loader_train = torch.utils.data.DataLoader(
        data_train ,
        batch_size=args.train.batch_size if not whole_audio else 1,
        shuffle=True,
        num_workers=args.train.num_workers if args.train.cache_device=='cpu' else 0,
        persistent_workers=(args.train.num_workers > 0) if args.train.cache_device=='cpu' else False,
        pin_memory=True if args.train.cache_device=='cpu' else False
    )
    data_valid = AudioDataset(
        args.data.valid_path,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=True,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
        use_spk=use_spk,
        use_spk_tokens=use_spk_tokens)
    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return loader_train, loader_valid 


class AudioDataset(Dataset):
    def __init__(
        self,
        path_root,
        waveform_sec,
        hop_size,
        sample_rate,
        load_all_data=True,
        whole_audio=False,
        extensions=['wav'],
        n_spk=1,
        device='cpu',
        fp16=False,
        use_aug=False,
        use_spk=False,
        use_spk_tokens=False,
        spk_cross_prob=0.5,
        spk_noise=0.02,
        spk_drop_prob=0.1,
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.crop_len = int(waveform_sec * sample_rate / hop_size)
        self.path_root = path_root
        self.paths = traverse_dir(
            os.path.join(path_root, 'audio'),
            extensions=extensions,
            is_pure=True,
            is_sort=True,
            is_ext=True
        )
        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.data_buffer={}
        self.pitch_aug_dict = np.load(os.path.join(self.path_root, 'pitch_aug_dict.npy'), allow_pickle=True).item()
        if load_all_data:
            print('Load all the data from :', path_root)
        else:
            print('Load the f0, volume data from :', path_root)

        # speaker embeddings (zero-shot timbre conditioning)
        self.use_spk = use_spk
        self.use_spk_tokens = use_spk_tokens
        self.spk_cross_prob = spk_cross_prob
        self.spk_noise = spk_noise
        self.spk_drop_prob = spk_drop_prob
        self.spk_buffer = {}
        self.spk_token_buffer = {}
        self.spk_to_indices = {}
        self.spk_dim = 0
        self.spk_token_dim = 0
        self.spk_token_dir = os.path.join(path_root, 'spk_tokens')
        if self.use_spk or self.use_spk_tokens:
            spk_dir = os.path.join(path_root, 'spk')
            missing = 0
            for idx, name_ext in enumerate(self.paths):
                speaker = os.path.dirname(name_ext)
                self.spk_to_indices.setdefault(speaker, []).append(idx)
                if self.use_spk:
                    path_spk = os.path.join(spk_dir, name_ext) + '.npy'
                    if os.path.exists(path_spk):
                        self.spk_buffer[name_ext] = np.load(path_spk).astype(np.float32)
                    else:
                        missing += 1
                if self.use_spk_tokens:
                    path_tok = os.path.join(self.spk_token_dir, name_ext) + '.npy'
                    if not os.path.exists(path_tok):
                        missing += 1
                    elif load_all_data:
                        self.spk_token_buffer[name_ext] = np.load(path_tok).astype(np.float32)
            if missing > 0:
                raise FileNotFoundError(
                    f' [x] {missing} speaker embedding/token files missing under {path_root}. '
                    f'Run preprocess.py or preprocess_spk.py first.')
            if self.use_spk:
                self.spk_dim = int(len(next(iter(self.spk_buffer.values()))))
            if self.use_spk_tokens:
                if len(self.spk_token_buffer) > 0:
                    _t = next(iter(self.spk_token_buffer.values()))
                else:
                    _first = self.paths[0]
                    _t = np.load(os.path.join(self.spk_token_dir, _first) + '.npy')
                self.spk_token_dim = int(_t.shape[-1])
                print(f'Load speaker tokens (n_tokens={_t.shape[0]}, dim={self.spk_token_dim}) from : {self.spk_token_dir}')
            print(f'Load {len(self.spk_buffer)} speaker embeddings (dim={self.spk_dim}, '
                  f'{len(self.spk_to_indices)} speakers) from : {spk_dir}')
        
        def _load_single_file(name_ext):
            name = os.path.splitext(name_ext)[0]

            path_f0 = os.path.join(self.path_root, 'f0', name_ext) + '.npy'
            f0 = np.load(path_f0)
            f0_len = len(f0)
            f0 = torch.from_numpy(f0).float().unsqueeze(-1).to(device)

            path_volume = os.path.join(self.path_root, 'volume', name_ext) + '.npy'
            volume = np.load(path_volume)
            volume_len = len(volume)
            volume = torch.from_numpy(volume).float().unsqueeze(-1).to(device)

            path_augvol = os.path.join(self.path_root, 'aug_vol', name_ext) + '.npy'
            aug_vol = np.load(path_augvol)
            aug_vol_len = len(aug_vol)
            aug_vol = torch.from_numpy(aug_vol).float().unsqueeze(-1).to(device)

            if n_spk is not None and n_spk > 1:
                dirname_split = re.split(r"_|\-", os.path.dirname(name_ext), 2)[0]
                spk_id = int(dirname_split) if str.isdigit(dirname_split) else 0
                if spk_id < 1 or spk_id > n_spk:
                    raise ValueError(' [x] Muiti-speaker training error : spk_id must be a positive integer from 1 to n_spk ')
            else:
                spk_id = 1
            spk_id = torch.LongTensor(np.array([spk_id])).to(device)

            path_mel = os.path.join(self.path_root, 'mel', name_ext) + '.npy'
            path_augmel = os.path.join(self.path_root, 'aug_mel', name_ext) + '.npy'
            path_units = os.path.join(self.path_root, 'units', name_ext) + '.npy'

            mel_len = get_npy_shape(path_mel)[0]
            aug_mel_len = get_npy_shape(path_augmel)[0]
            units_len = get_npy_shape(path_units)[0]
            frame_len = min(mel_len, aug_mel_len, units_len, f0_len, volume_len, aug_vol_len)

            if load_all_data:
                mel = np.load(path_mel)
                mel = torch.from_numpy(mel).to(device)

                aug_mel = np.load(path_augmel)
                aug_mel = torch.from_numpy(aug_mel).to(device)

                units = np.load(path_units)
                units = torch.from_numpy(units).to(device)

                if fp16:
                    mel = mel.half()
                    aug_mel = aug_mel.half()
                    units = units.half()

                data_dict = {
                        'frame_len': frame_len,
                        'mel': mel,
                        'aug_mel': aug_mel,
                        'units': units,
                        'f0': f0,
                        'volume': volume,
                        'aug_vol': aug_vol,
                        'spk_id': spk_id
                        }
            else:
                data_dict = {
                        'frame_len': frame_len,
                        'f0': f0,
                        'volume': volume,
                        'aug_vol': aug_vol,
                        'spk_id': spk_id
                        }
            return name_ext, data_dict

        max_workers = max(min(32, os.cpu_count()), 4)
        print(f'Using {max_workers} workers for parallel data loading')
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {executor.submit(_load_single_file, name_ext): name_ext for name_ext in self.paths}
            for future in tqdm(concurrent.futures.as_completed(future_to_name), total=len(self.paths), desc='Loading data'):
                try:
                    name_ext, data_dict = future.result()
                    self.data_buffer[name_ext] = data_dict
                except Exception as e:
                    print(f'Error loading {future_to_name[future]}: {e}')
                    raise

    def __getitem__(self, file_idx):
        name_ext = self.paths[file_idx]
        data_buffer = self.data_buffer[name_ext]
        # check duration. if too short, then skip
        if data_buffer['frame_len'] < self.crop_len:
            return self.__getitem__( (file_idx + 1) % len(self.paths))
            
        # get item
        return self.get_data(name_ext, data_buffer)

    def _load_spk_tokens(self, name_ext):
        buf = self.spk_token_buffer.get(name_ext)
        if buf is not None:
            return torch.from_numpy(np.array(buf, dtype=np.float32, copy=True))
        path_tok = os.path.join(self.spk_token_dir, name_ext) + '.npy'
        return torch.from_numpy(np.load(path_tok).astype(np.float32))

    def get_data(self, name_ext, data_buffer):
        name = os.path.splitext(name_ext)[0]
        start_frame = 0 if self.whole_audio else random.randint(0, data_buffer['frame_len'] - self.crop_len)
        units_frame_len = data_buffer['frame_len'] if self.whole_audio else self.crop_len
        aug_flag = random.choice([True, False]) and self.use_aug

        # load mel
        mel_key = 'aug_mel' if aug_flag else 'mel'
        mel = data_buffer.get(mel_key)
        if mel is None:
            mel = os.path.join(self.path_root, mel_key, name_ext) + '.npy'
            mel = np.load(mel, mmap_mode='r')
            mel = mel[start_frame : start_frame + units_frame_len].copy()
            mel = torch.from_numpy(mel).float() 
        else:
            mel = mel[start_frame : start_frame + units_frame_len]
            
        # load units
        units = data_buffer.get('units')
        if units is None:
            units = os.path.join(self.path_root, 'units', name_ext) + '.npy'
            units = np.load(units, mmap_mode='r')
            units = units[start_frame : start_frame + units_frame_len].copy()
            units = torch.from_numpy(units).float() 
        else:
            units = units[start_frame : start_frame + units_frame_len]

        # load f0
        f0 = data_buffer.get('f0')
        aug_shift = 0
        if aug_flag:
            aug_shift = self.pitch_aug_dict[name_ext]
        f0_frames = 2 ** (aug_shift / 12) * f0[start_frame : start_frame + units_frame_len]
        
        # load volume
        vol_key = 'aug_vol' if aug_flag else 'volume'
        volume = data_buffer.get(vol_key)
        volume_frames = volume[start_frame : start_frame + units_frame_len]
        
        # load spk_id
        spk_id = data_buffer.get('spk_id')
        
        # load shift
        aug_shift = torch.from_numpy(np.array([[aug_shift]])).float()

        # speaker embedding / tokens (cross-utterance pairing + noise/dropout to avoid content leakage)
        spk_vec = None
        spk_tokens = None
        if self.use_spk or self.use_spk_tokens:
            ref_name = name_ext
            if self.use_aug and random.random() < self.spk_cross_prob:
                candidates = self.spk_to_indices.get(os.path.dirname(name_ext), [])
                if len(candidates) > 1:
                    while ref_name == name_ext:
                        ref_name = self.paths[random.choice(candidates)]
            drop = self.use_aug and self.spk_drop_prob > 0 and random.random() < self.spk_drop_prob
            if self.use_spk:
                z = np.array(self.spk_buffer[ref_name], dtype=np.float32, copy=True)
                if self.use_aug and self.spk_noise > 0:
                    z = z + np.random.randn(*z.shape).astype(np.float32) * self.spk_noise
                if drop:
                    z = np.zeros_like(z)
                z = z / (np.linalg.norm(z) + 1e-6)
                spk_vec = torch.from_numpy(z.astype(np.float32))
            if self.use_spk_tokens:
                tok = self._load_spk_tokens(ref_name)
                if self.use_aug and self.spk_noise > 0:
                    tok = tok + torch.randn_like(tok) * self.spk_noise
                if drop:
                    tok = torch.zeros_like(tok)
                spk_tokens = tok

        data = dict(mel=mel, f0=f0_frames, volume=volume_frames, units=units,
                    spk_id=spk_id, aug_shift=aug_shift, name=name, name_ext=name_ext)
        if spk_vec is not None:
            data['spk_vec'] = spk_vec
        if spk_tokens is not None:
            data['spk_tokens'] = spk_tokens
        return data

    def __len__(self):
        return len(self.paths)
