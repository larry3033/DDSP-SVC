import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

DEFAULT_SPK_MODEL = 'microsoft/wavlm-base-plus-sv'


def _bypass_torch_load_gate_once():
    """Allow loading trusted `.bin` weights on torch < 2.6.

    transformers >= 4.51 refuses `torch.load` when torch < 2.6 (CVE-2025-32434).
    `microsoft/wavlm-base-plus-sv` ships only `.bin` weights, so on older torch we
    bypass that version gate. Prefer upgrading torch to >= 2.6 if you can.
    """
    patched = False
    for mod_name in ('transformers.modeling_utils', 'transformers.utils.import_utils'):
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            if hasattr(mod, 'check_torch_load_is_safe'):
                mod.check_torch_load_is_safe = lambda *a, **k: None
                patched = True
        except Exception:
            pass
    return patched


def _from_pretrained_with_fallback(cls, model_id):
    try:
        return cls.from_pretrained(model_id)
    except ValueError as e:
        msg = str(e)
        if 'torch' in msg and 'v2.6' in msg and 'safetensors' in msg:
            print(' [Spk Encoder] torch < 2.6 detected: bypassing transformers torch.load gate '
                  '(recommended: upgrade torch to >= 2.6)')
            _bypass_torch_load_gate_once()
            return cls.from_pretrained(model_id)
        raise



class SpkEncoder:
    """Extract a fixed-length, content-invariant speaker embedding from a reference waveform.

    Backed by a frozen pretrained speaker-verification model (WavLM x-vector).
    The output is L2-normalized so it can be injected as a global speaker condition.
    """

    def __init__(self, backend='wavlm-sv', model_id=None, device='cpu', sample_rate=16000):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.sample_rate = sample_rate
        self.backend = backend
        self.resample_kernel = {}

        if backend == 'wavlm-sv':
            from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor
            self.model_id = model_id or DEFAULT_SPK_MODEL
            print(' [Spk Encoder] WavLM x-vector')
            print(' [Loading] ' + self.model_id)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_id)
            self.model = _from_pretrained_with_fallback(WavLMForXVector, self.model_id).to(self.device)
            self.model.eval()
            self.dim = int(getattr(self.model.config, 'xvector_output_dim', 512))
            self.hidden_dim = int(getattr(self.model.config, 'hidden_size', 768))
        else:
            raise ValueError(f" [x] Unknown speaker encoder: {backend}")

    def _to_tensor(self, audio, sample_rate):
        if isinstance(audio, np.ndarray):
            wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).float()
        else:
            wav = audio.detach().float().reshape(-1).cpu()
        if wav.dim() > 1:
            wav = wav.mean(dim=0)
        if sample_rate != self.sample_rate:
            key = str(sample_rate)
            if key not in self.resample_kernel:
                self.resample_kernel[key] = Resample(
                    sample_rate, self.sample_rate, lowpass_filter_width=128).to(self.device)
            wav = self.resample_kernel[key](wav.unsqueeze(0).to(self.device)).squeeze(0)
        else:
            wav = wav.to(self.device)
        if wav.numel() < self.sample_rate // 4:
            wav = F.pad(wav, (0, self.sample_rate // 4 - wav.numel()))
        return wav

    @torch.no_grad()
    def encode(self, audio, sample_rate=16000, normalize=True):
        """audio: 1d numpy array or 1d/2d torch tensor. Returns a 1d numpy float32 vector."""
        wav = self._to_tensor(audio, sample_rate)
        inputs = self.feature_extractor(
            wav.cpu().numpy(), sampling_rate=self.sample_rate, return_tensors='pt').input_values.to(self.device)
        emb = self.model(inputs).embeddings
        if emb.dim() > 2:
            emb = emb.reshape(emb.shape[0], -1)
        emb = emb.squeeze(0).float()
        if normalize:
            emb = F.normalize(emb, dim=-1, eps=1e-6)
        return emb.cpu().numpy().astype(np.float32)

    def encode_files(self, paths, normalize=True):
        """Average the (normalized) embeddings of several reference files into a single vector."""
        vecs = []
        for p in paths:
            import librosa
            audio, sr = librosa.load(p, sr=self.sample_rate, mono=True)
            vecs.append(self.encode(audio, sr, normalize=True))
        z = np.mean(np.stack(vecs, axis=0), axis=0)
        if normalize:
            n = np.linalg.norm(z) + 1e-6
            z = z / n
        return z.astype(np.float32)

    @torch.no_grad()
    def encode_tokens(self, audio, sample_rate=16000, n_tokens=16, normalize=True):
        """Frame-level speaker tokens: adaptive-pool WavLM hidden states to `n_tokens`.

        Returns a 2d numpy float32 array of shape [n_tokens, hidden_dim].
        """
        wav = self._to_tensor(audio, sample_rate)
        inputs = self.feature_extractor(
            wav.cpu().numpy(), sampling_rate=self.sample_rate, return_tensors='pt').input_values.to(self.device)
        hidden = self.model.wavlm(inputs).last_hidden_state  # [1, T', H]
        hidden = hidden.transpose(1, 2)  # [1, H, T']
        if hidden.shape[-1] >= n_tokens:
            tokens = F.adaptive_avg_pool1d(hidden, n_tokens)
        else:
            tokens = F.interpolate(hidden, size=n_tokens, mode='linear', align_corners=False)
        tokens = tokens.transpose(1, 2).squeeze(0).float()  # [K, H]
        if normalize:
            tokens = F.normalize(tokens, dim=-1, eps=1e-6)
        return tokens.cpu().numpy().astype(np.float32)

    def encode_tokens_files(self, paths, n_tokens=16, normalize=True):
        """Concatenate the token sequences of several reference files: [K*n_ref, hidden_dim]."""
        toks = []
        for p in paths:
            import librosa
            audio, sr = librosa.load(p, sr=self.sample_rate, mono=True)
            toks.append(self.encode_tokens(audio, sr, n_tokens=n_tokens, normalize=normalize))
        return np.concatenate(toks, axis=0).astype(np.float32)
