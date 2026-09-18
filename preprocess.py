import os
import numpy as np
import random
import librosa
import torch
import argparse
import shutil
import concurrent.futures
import torch.multiprocessing as mp
from logger import utils
from tqdm import tqdm
from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
from reflow.vocoder import Vocoder
from logger.utils import traverse_dir


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set")
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=2,
        required=False,
        help="number of worker processes (default: cpu_count)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="reprocess every file, ignoring existing outputs (disables resume)")
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="truncate clips longer than this many seconds (0 = no limit) | default: 30")
    return parser.parse_args(args=args, namespace=namespace)


def load_extractors(args, sample_rate=None, hop_size=None, device='cuda'):
    if sample_rate is None:
        sample_rate = args.data.sampling_rate
    if hop_size is None:
        hop_size = args.data.block_size
    
    # initialize f0 extractor
    f0_extractor = F0_Extractor(
                        args.data.f0_extractor, 
                        sample_rate, 
                        hop_size, 
                        args.data.f0_min, 
                        args.data.f0_max)
    
    # initialize volume extractor
    volume_extractor = Volume_Extractor(hop_size, args.data.volume_smooth_size)
    
    # initialize mel extractor
    mel_extractor = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
    if mel_extractor.vocoder_sample_rate != sample_rate or mel_extractor.vocoder_hop_size != hop_size:
        mel_extractor = None
        print('Unmatch vocoder parameters, mel extraction is ignored!')
    
    # initialize units encoder
    units_encoder = Units_Encoder(
                        args.data.encoder, 
                        args.data.encoder_ckpt, 
                        args.data.encoder_sample_rate, 
                        args.data.encoder_hop_size,
                        device=device)

    # initialize speaker encoder (optional, for zero-shot timbre conditioning)
    spk_encoder = None
    if args.model.get('use_spk', False):
        from encoder.spk_encoder import SpkEncoder
        spk_encoder = SpkEncoder(
                        args.model.get('spk_backend', 'wavlm-sv'),
                        args.model.get('spk_model', None),
                        device=device)
    return f0_extractor, volume_extractor, mel_extractor, units_encoder, spk_encoder
    
    
# Global variables for worker processes (each process has its own copy)
_worker_f0_extractor = None
_worker_volume_extractor = None
_worker_mel_extractor = None
_worker_units_encoder = None
_worker_spk_encoder = None
_worker_device = None


def _worker_init(args, device):
    """Initialize extractors for each worker process."""
    global _worker_f0_extractor, _worker_volume_extractor, _worker_mel_extractor, _worker_units_encoder, _worker_spk_encoder, _worker_device
    _worker_device = device
    _worker_f0_extractor, _worker_volume_extractor, _worker_mel_extractor, _worker_units_encoder, _worker_spk_encoder = load_extractors(
        args, device=device)


def _process_file(file, path, sample_rate, hop_size, use_pitch_aug, extensions, use_spk_tokens=False, spk_n_tokens=16, max_duration=0):
    """
    Process a single audio file using either provided extractors or worker-global extractors.

    Args:
        file: Audio filename
        path: Base directory path
        sample_rate, hop_size: Audio parameters
        use_pitch_aug: Whether to use pitch augmentation
        extensions: Audio file extensions
        f0_extractor, volume_extractor, mel_extractor, units_encoder: Optional extractors
        device: Optional processing device

    Returns:
        keyshift value if successful, None otherwise
    """
    global _worker_f0_extractor, _worker_volume_extractor, _worker_mel_extractor, _worker_units_encoder, _worker_spk_encoder, _worker_device

    # Use provided extractors if available, otherwise use global ones
    f0_extractor = _worker_f0_extractor
    volume_extractor = _worker_volume_extractor
    mel_extractor = _worker_mel_extractor
    units_encoder = _worker_units_encoder
    spk_encoder = _worker_spk_encoder
    device = _worker_device

    path_srcdir  = os.path.join(path, 'audio')
    path_unitsdir  = os.path.join(path, 'units')
    path_f0dir  = os.path.join(path, 'f0')
    path_volumedir  = os.path.join(path, 'volume')
    path_augvoldir  = os.path.join(path, 'aug_vol')
    path_meldir  = os.path.join(path, 'mel')
    path_augmeldir  = os.path.join(path, 'aug_mel')
    path_spkdir = os.path.join(path, 'spk')
    path_spktokdir = os.path.join(path, 'spk_tokens')
    path_skipdir = os.path.join(path, 'skip')

    binfile = file+'.npy'
    path_srcfile = os.path.join(path_srcdir, file)
    path_unitsfile = os.path.join(path_unitsdir, binfile)
    path_f0file = os.path.join(path_f0dir, binfile)
    path_volumefile = os.path.join(path_volumedir, binfile)
    path_augvolfile = os.path.join(path_augvoldir, binfile)
    path_melfile = os.path.join(path_meldir, binfile)
    path_augmelfile = os.path.join(path_augmeldir, binfile)
    path_spkfile = os.path.join(path_spkdir, binfile)
    path_spktokfile = os.path.join(path_spktokdir, binfile)
    path_skipfile = os.path.join(path_skipdir, file)
    
    # load audio
    audio, _ = librosa.load(path_srcfile, sr=sample_rate)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    # bound memory: truncate overly long clips (all outputs derive from this audio)
    if max_duration and max_duration > 0:
        max_len = int(max_duration * sample_rate)
        if len(audio) > max_len:
            audio = audio[:max_len]
    audio_t = torch.from_numpy(audio).float().to(device)
    audio_t = audio_t.unsqueeze(0)
    
    # extract volume
    volume = volume_extractor.extract(audio)
    
    # extract mel and volume augmentaion
    if mel_extractor is not None:
        mel_t = mel_extractor.extract(audio_t, sample_rate)
        mel = mel_t.squeeze().to('cpu').numpy()
        
        max_amp = float(torch.max(torch.abs(audio_t))) + 1e-5
        max_shift = min(1, np.log10(1/max_amp))
        log10_vol_shift = random.uniform(-1, max_shift)
        if use_pitch_aug:
            keyshift = random.uniform(-5, 5)
        else:
            keyshift = 0
        
        aug_mel_t = mel_extractor.extract(audio_t * (10 ** log10_vol_shift), sample_rate, keyshift = keyshift)
        aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
        aug_vol = volume_extractor.extract(audio * (10 ** log10_vol_shift))
        
    # units encode
    units_t = units_encoder.encode(audio_t, sample_rate, hop_size)
    units = units_t.squeeze().to('cpu').numpy()
    
    # extract f0
    f0 = f0_extractor.extract(audio, uv_interp = False)
    
    uv = f0 == 0
    if len(f0[~uv]) > 0:
        # interpolate the unvoiced f0
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

        # save npy     
        os.makedirs(os.path.dirname(path_unitsfile), exist_ok=True)
        np.save(path_unitsfile, units.astype(np.float16))
        os.makedirs(os.path.dirname(path_f0file), exist_ok=True)
        np.save(path_f0file, f0.astype(np.float32))
        os.makedirs(os.path.dirname(path_volumefile), exist_ok=True)
        np.save(path_volumefile, volume)
        if mel_extractor is not None:
            os.makedirs(os.path.dirname(path_melfile), exist_ok=True)
            np.save(path_melfile, np.ascontiguousarray(mel).astype(np.float16))
            os.makedirs(os.path.dirname(path_augmelfile), exist_ok=True)
            np.save(path_augmelfile, np.ascontiguousarray(aug_mel).astype(np.float16))
            os.makedirs(os.path.dirname(path_augvolfile), exist_ok=True)
            np.save(path_augvolfile, aug_vol)

        # speaker embedding (for zero-shot timbre conditioning)
        if spk_encoder is not None:
            spk = spk_encoder.encode(audio, sample_rate)
            os.makedirs(os.path.dirname(path_spkfile), exist_ok=True)
            np.save(path_spkfile, np.asarray(spk, dtype=np.float32))
            # speaker tokens (for cross-attention conditioning)
            if use_spk_tokens:
                toks = spk_encoder.encode_tokens(audio, sample_rate, n_tokens=spk_n_tokens)
                os.makedirs(os.path.dirname(path_spktokfile), exist_ok=True)
                np.save(path_spktokfile, np.asarray(toks, dtype=np.float16))

        return keyshift
    else:
        print('\n[Error] F0 extraction failed: ' + path_srcfile)
        os.makedirs(os.path.dirname(path_skipfile), exist_ok=True)
        shutil.move(path_srcfile, os.path.dirname(path_skipfile))
        print('This file has been moved to ' + path_skipfile)
    return None


def _outputs_done(path, file, use_mel=True, use_spk=False, use_spk_tokens=False, pitch_aug_dict=None):
    """Return True if every expected output for `file` already exists (for --resume).

    Also requires a pitch-augmentation entry, otherwise training would `KeyError`
    on `self.pitch_aug_dict[name_ext]` (aug entries are only written by workers).
    """
    binfile = file + '.npy'
    needed = [
        os.path.join(path, 'units', binfile),
        os.path.join(path, 'f0', binfile),
        os.path.join(path, 'volume', binfile),
    ]
    if use_mel:
        needed += [
            os.path.join(path, 'mel', binfile),
            os.path.join(path, 'aug_mel', binfile),
            os.path.join(path, 'aug_vol', binfile),
        ]
    if use_spk:
        needed.append(os.path.join(path, 'spk', binfile))
    if use_spk_tokens:
        needed.append(os.path.join(path, 'spk_tokens', binfile))
    if pitch_aug_dict is not None and file not in pitch_aug_dict:
        return False
    return all(os.path.exists(p) for p in needed)


def preprocess(path, args, sample_rate=None, hop_size=None, device='cuda', use_pitch_aug=False, extensions=['wav'], workers=1, use_spk_tokens=False, spk_n_tokens=16, use_spk=False, use_mel=True, resume=True, max_duration=0):
    # List files
    path_srcdir = os.path.join(path, 'audio')
    filelist = traverse_dir(
        path_srcdir,
        extensions=extensions,
        is_pure=True,
        is_sort=True,
        is_ext=True)

    # Ensure sample_rate and hop_size are set
    if sample_rate is None:
        sample_rate = args.data.sampling_rate
    if hop_size is None:
        hop_size = args.data.block_size

    # keep the existing pitch-augmentation dict so already-processed files stay consistent
    path_pitchaugdict = os.path.join(path, 'pitch_aug_dict.npy')
    pitch_aug_dict = {}
    if resume and os.path.exists(path_pitchaugdict):
        try:
            pitch_aug_dict = dict(np.load(path_pitchaugdict, allow_pickle=True).item())
        except Exception:
            pitch_aug_dict = {}

    # resume: skip files whose outputs AND pitch-aug entry already exist
    if resume:
        pending = [f for f in filelist
                   if not _outputs_done(path, f, use_mel, use_spk, use_spk_tokens, pitch_aug_dict)]
        print(f'[resume] {path}: {len(filelist) - len(pending)} already done, {len(pending)} to process')
    else:
        pending = list(filelist)

    # Prepare arguments for worker initialization
    init_args = (args, device)

    # Multiprocessing with ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=init_args
    ) as executor:
        # Submit tasks
        future_to_file = {
            executor.submit(_process_file, file, path, sample_rate, hop_size, use_pitch_aug, extensions, use_spk_tokens, spk_n_tokens, max_duration): file
            for file in pending
        }

        # Collect results with progress bar
        flush_every = 2000
        finished = 0
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(pending)):
            file = future_to_file[future]
            try:
                keyshift = future.result()
                if keyshift is not None:
                    pitch_aug_dict[file] = keyshift
            except Exception as e:
                print(f'\n[Error] Task for {file} generated an exception: {e}')
            finished += 1
            # periodically flush so a crash doesn't lose the pitch-aug entries
            if len(pitch_aug_dict) > 0 and finished % flush_every == 0:
                np.save(path_pitchaugdict, pitch_aug_dict)

    # Save pitch augmentation dictionary if any
    if len(pitch_aug_dict) > 0:
        np.save(path_pitchaugdict, pitch_aug_dict)
        print(f'Saved pitch augmentation dictionary to {path_pitchaugdict}')

    
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    # parse commands
    cmd = parse_args()

    device = cmd.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load config
    args = utils.load_config(cmd.config)

    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size
    extensions = args.data.extensions
    train_path = args.data.train_path
    valid_path = args.data.valid_path
    use_pitch_aug = args.model.use_pitch_aug
    use_spk = bool(args.model.get('use_spk', False))
    use_spk_tokens = bool(args.model.get('use_spk_tokens', False)) if use_spk else False
    spk_n_tokens = int(args.model.get('spk_n_tokens', 16))
    resume = not cmd.force
    
    # get number of workers
    workers = cmd.workers
    if workers is None:
        import multiprocessing
        workers = multiprocessing.cpu_count()
    print(f'Using {workers} worker processes')   

    # parallel processing
    preprocess(train_path,
               args=args,
               sample_rate=sample_rate,
               hop_size=hop_size,
               device=device,
               use_pitch_aug=use_pitch_aug,
               extensions=extensions,
               workers=workers,
               use_spk_tokens=use_spk_tokens,
               spk_n_tokens=spk_n_tokens,
               use_spk=use_spk,
               resume=resume,
               max_duration=cmd.max_duration)

    # preprocess validation set (no pitch augmentation)
    preprocess(valid_path,
               args=args,
               sample_rate=sample_rate,
               hop_size=hop_size,
               device=device,
               use_pitch_aug=False,
               extensions=extensions,
               workers=workers,
               use_spk_tokens=use_spk_tokens,
               spk_n_tokens=spk_n_tokens,
               use_spk=use_spk,
               resume=resume,
               max_duration=cmd.max_duration)