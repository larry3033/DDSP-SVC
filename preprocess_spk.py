import os
import argparse
import numpy as np
import librosa
from tqdm import tqdm
from logger import utils
from logger.utils import traverse_dir
from encoder.spk_encoder import SpkEncoder


def extract_spk(path, args, encoder, extensions, use_spk_tokens=False, n_tokens=16):
    srcdir = os.path.join(path, 'audio')
    outdir = os.path.join(path, 'spk')
    tokdir = os.path.join(path, 'spk_tokens')
    filelist = traverse_dir(
        srcdir,
        extensions=extensions,
        is_pure=True,
        is_sort=True,
        is_ext=True)

    if len(filelist) == 0:
        print(f' [x] no audio found under {srcdir}')
        return

    skipped = 0
    extracted = 0
    for file in tqdm(filelist, desc=f'spk: {path}'):
        out_path = os.path.join(outdir, file) + '.npy'
        tok_path = os.path.join(tokdir, file) + '.npy'
        need_vec = not os.path.exists(out_path)
        need_tok = use_spk_tokens and not os.path.exists(tok_path)
        if not need_vec and not need_tok:
            skipped += 1
            continue
        audio, sr = librosa.load(os.path.join(srcdir, file), sr=encoder.sample_rate, mono=True)
        if need_vec:
            z = encoder.encode(audio, sr)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, z.astype(np.float32))
        if need_tok:
            toks = encoder.encode_tokens(audio, sr, n_tokens=n_tokens)
            os.makedirs(os.path.dirname(tok_path), exist_ok=True)
            np.save(tok_path, toks.astype(np.float16))
        extracted += 1
    print(f' [ok] {path}: {extracted} extracted, {skipped} already existed')


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="path to the config file")
    parser.add_argument("-d", "--device", type=str, default=None, required=False,
                        help="cpu or cuda, auto if not set")
    parser.add_argument("--force", action="store_true", help="recompute even if `spk/*.npy` exists")
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    import torch
    cmd = parse_args()

    device = cmd.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = utils.load_config(cmd.config)
    backend = args.model.get('spk_backend', 'wavlm-sv')
    model_id = args.model.get('spk_model', None)
    encoder = SpkEncoder(backend=backend, model_id=model_id, device=device)
    print(f' [Spk Encoder] dim = {encoder.dim}, token_dim = {encoder.hidden_dim}')

    use_spk_tokens = bool(args.model.get('use_spk_tokens', False)) if args.model.get('use_spk', False) else False
    n_tokens = int(args.model.get('spk_n_tokens', 16))

    if cmd.force:
        for base in [args.data.train_path, args.data.valid_path]:
            for sub in ['spk', 'spk_tokens']:
                outdir = os.path.join(base, sub)
                if os.path.isdir(outdir):
                    for root, _, files in os.walk(outdir):
                        for f in files:
                            os.remove(os.path.join(root, f))

    for base in [args.data.train_path, args.data.valid_path]:
        extract_spk(base, args, encoder, args.data.extensions, use_spk_tokens, n_tokens)
