import numpy as np
import torch
import torch.nn.functional as F
from transformers import HubertModel, HubertConfig
from torchaudio.transforms import Resample
from .unit2control import Unit2Control

F0_KERNEL = {}

class F0_Extractor:
    def __init__(self, f0_extractor, sample_rate = 44100, hop_size = 512, f0_min = 65, f0_max = 800):
        self.f0_extractor = f0_extractor
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.f0_min = f0_min
        self.f0_max = f0_max
        if f0_extractor == 'rmvpe':
            if 'rmvpe' not in F0_KERNEL :
                from encoder.rmvpe import RMVPE
                F0_KERNEL['rmvpe'] = RMVPE('pretrain/rmvpe/model.pt', hop_length=160)
            self.rmvpe = F0_KERNEL['rmvpe']
        else:
            raise ValueError(f" [x] Unknown f0 extractor: {f0_extractor}")
                
    def extract(self, audio, uv_interp = False, device = None, silence_front = 0): # audio: 1d numpy array
        # extractor start time
        n_frames = int(len(audio) // self.hop_size) + 1
                
        start_frame = int(silence_front * self.sample_rate / self.hop_size)
        real_silence_front = start_frame * self.hop_size / self.sample_rate
        audio = audio[int(np.round(real_silence_front * self.sample_rate)) : ]
        
        # extract f0 using rmvpe
        f0 = self.rmvpe.infer_from_audio(audio, self.sample_rate, device=device, thred=0.03, use_viterbi=False)
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        origin_time = 0.01 * np.arange(len(f0))
        target_time = self.hop_size / self.sample_rate * np.arange(n_frames - start_frame)
        f0 = np.interp(target_time, origin_time, f0)
        uv = np.interp(target_time, origin_time, uv.astype(float)) > 0.5
        f0[uv] = 0
        f0 = np.pad(f0, (start_frame, 0))
                    
        # interpolate the unvoiced f0 
        if uv_interp:
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            f0[f0 < self.f0_min] = self.f0_min
        return f0


class Volume_Extractor:
    def __init__(self, hop_size = 512, win_size = 2048):
        self.hop_size = hop_size
        self.win_size = win_size
        
    def extract(self, audio): # audio: 1d numpy array
        n_frames = int(len(audio) // self.hop_size) + 1
        audio = np.pad(audio, (int(self.win_size // 2), int((self.win_size + 1) // 2)), mode = 'reflect')
        audio2 = audio ** 2
        mean = np.array([np.mean(audio[int(n * self.hop_size) : int(n * self.hop_size + self.win_size)]) for n in range(n_frames)])
        mean_square = np.array([np.mean(audio2[int(n * self.hop_size) : int(n * self.hop_size + self.win_size)]) for n in range(n_frames)])
        volume = np.sqrt(np.clip(mean_square - mean ** 2, 0, None))
        return volume
    
         
class Units_Encoder:
    def __init__(self, encoder, encoder_ckpt, encoder_sample_rate = 16000, encoder_hop_size = 160, device = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        if encoder == 'contentvec768l12tta2x':
            self.model = Audio2ContentVec768L12TTA2X(encoder_ckpt, device=device)
        else:
            raise ValueError(f" [x] Unknown units encoder: {encoder}")
            
        self.resample_kernel = {}
        self.encoder_sample_rate = encoder_sample_rate
        self.encoder_hop_size = encoder_hop_size
        
    def encode(self, 
                audio, # B, T
                sample_rate,
                hop_size): 
        
        # resample
        if sample_rate == self.encoder_sample_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.encoder_sample_rate, lowpass_filter_width = 128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)
        
        # encode
        if audio_res.size(-1) < 400:
            audio_res = torch.nn.functional.pad(audio_res, (0, 400 - audio_res.size(-1)))
        units = self.model(audio_res)
        
        # alignment
        n_frames = audio.size(-1) // hop_size + 1
        ratio = (hop_size / sample_rate) / (self.encoder_hop_size / self.encoder_sample_rate)
        index = torch.clamp(torch.round(ratio * torch.arange(n_frames).to(self.device)).long(), max = units.size(1) - 1)
        units_aligned = torch.gather(units, 1, index.unsqueeze(0).unsqueeze(-1).repeat([1, 1, units.size(-1)]))
        return units_aligned


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = torch.nn.Linear(config.hidden_size, config.classifier_proj_size)
        
         
class Audio2ContentVec768L12TTA2X():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=160, device='cpu'):
        self.device = device
        print(' [Encoder Model] Content Vec')
        print(' [Loading] ' + path)
        self.hubert = HubertModelWithFinalProj(HubertConfig())
        checkpoint = torch.load(path)
        self.hubert.load_state_dict(checkpoint)
        self.hubert = self.hubert.to(self.device)
        self.hubert.eval()

    def __call__(self,
                 audio):  # B, T
        with torch.no_grad():
            feats = self.hubert(audio)["last_hidden_state"]
            audio = F.pad(audio, (160, 0))
            feats2 = self.hubert(audio)["last_hidden_state"]
            n = feats2.shape[1] - feats.shape[1]
            if n > 0:
                feats = F.pad(feats, (0, 0, 0, 1))
            feats_tta = torch.cat((feats2, feats), dim=2).reshape(feats.shape[0], -1, feats.shape[-1])
            feats_tta = feats_tta[:, 1:, :]
            if n > 0:
                feats_tta = feats_tta[:, :-1, :]
        return feats_tta


class DotDict(dict):
    def __getattr__(self, key):
        if key.startswith('__') and key.endswith('__'):
            raise AttributeError(key)
        val = dict.get(self, key)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__

    
class CombSubSuperFast(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            win_length,
            n_unit=256,
            n_spk=1,
            num_layers=3,
            dim_model=256,
            use_norm=False,
            use_attention=False,
            use_pitch_aug=False,
            spk_dim=0,
            use_spk_film=False,
            spk_token_dim=0):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("win_length", torch.tensor(win_length))
        self.register_buffer("window", torch.hann_window(win_length))
        #Unit2Control
        split_map = {
            'harmonic_magnitude': win_length // 2 + 1, 
            'harmonic_phase': win_length // 2 + 1,
            'noise_magnitude': win_length // 2 + 1,
            'noise_phase': win_length // 2 + 1
        }
        self.unit2ctrl = Unit2Control(
                            n_unit, 
                            block_size, 
                            n_spk, 
                            split_map,
                            num_layers=num_layers,
                            dim_model=dim_model,
                            use_norm=use_norm,
                            use_attention=use_attention, 
                            use_pitch_aug=use_pitch_aug,
                            spk_dim=spk_dim,
                            use_spk_film=use_spk_film,
                            spk_token_dim=spk_token_dim)
    
    def fast_source_gen(self, f0_frames):
        n = torch.arange(self.block_size, device=f0_frames.device)
        s0 = f0_frames / self.sampling_rate
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * (n + 1) + 0.5 * ds0 * n * (n + 1) / self.block_size
        s0 = s0 + ds0 * n / self.block_size
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0_frames)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad -= torch.round(rad)
        combtooth = torch.sinc(rad / (s0 + 1e-5)).reshape(f0_frames.shape[0], -1)
        return combtooth
        
    def forward(self, units_frames, f0_frames, volume_frames, spk_id=None, spk_mix_dict=None, aug_shift=None, spk_vec=None, spk_tokens=None, initial_phase=None, infer=True, **kwargs):
        '''
            units_frames: B x n_frames x n_unit
            f0_frames: B x n_frames x 1
            volume_frames: B x n_frames x 1 
            spk_id: B x 1
            spk_vec: B x spk_dim (continuous speaker embedding)
            spk_tokens: B x n_tokens x spk_token_dim (speaker tokens for cross-attention)
        '''
        # combtooth exciter signal
        combtooth = self.fast_source_gen(f0_frames)
        combtooth_frames = combtooth.unfold(1, self.block_size, self.block_size)
        
        # noise exciter signal
        noise = torch.randn_like(combtooth)
        noise_frames = noise.unfold(1, self.block_size, self.block_size)
        
        # parameter prediction
        ctrls, hidden = self.unit2ctrl(units_frames, combtooth_frames, noise_frames, volume_frames, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift, spk_vec=spk_vec, spk_tokens=spk_tokens)
        
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude'] + 1.j * np.pi * ctrls['noise_phase']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # harmonic part filter
        if combtooth.shape[-1] > self.win_length // 2:
            pad_mode = 'reflect'
        else:
            pad_mode = 'constant'
        combtooth_stft = torch.stft(
                            combtooth,
                            n_fft = self.win_length,
                            win_length = self.win_length,
                            hop_length = self.block_size,
                            window = self.window,
                            center = True,
                            return_complex = True,
                            pad_mode = pad_mode)
        
        # noise part filter
        noise_stft = torch.stft(
                            noise,
                            n_fft = self.win_length,
                            win_length = self.win_length,
                            hop_length = self.block_size,
                            window = self.window,
                            center = True,
                            return_complex = True,
                            pad_mode = pad_mode)
        
        # apply the filters 
        signal_stft = combtooth_stft * src_filter.permute(0, 2, 1) + noise_stft * noise_filter.permute(0, 2, 1)
        
        # take the istft to resynthesize audio.
        signal = torch.istft(
                        signal_stft,
                        n_fft = self.win_length,
                        win_length = self.win_length,
                        hop_length = self.block_size,
                        window = self.window,
                        center = True)

        return signal, hidden
