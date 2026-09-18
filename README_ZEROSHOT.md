# DDSP-SVC 6.3 — Zero-shot 改造 详细说明（交接文档）

> 本文档是本次改造的完整交接说明，供后续会话/人员直接接手。
> 分支：fork `larry3033/DDSP-SVC` 的 **`zeroshot`** 分支，基于上游 `yxlllc/DDSP-SVC` 的 `6.3` 分支。
> 本地开发目录：`D:\DDSP-SVC-6.3`（**非 git 仓库**，GitHub 上见上述仓库）。
> 配套的项目记忆/操作速查见同目录 `AGENTS.md`。
> 原始上游 README 保留在 `README.md`。

---

## 0. TL;DR

- 在 **DDSP-SVC 6.3（RectifiedFlow 版）** 上实现了 **zero-shot 音色克隆**：给定一段参考音频，即可把任意干声转换成参考说话人的音色。
- 方案为 **C 档**：`全局说话人向量(WavLM-SV 512) + DDSP 逐层 FiLM + reflow 逐 block FiLM + 说话人 token 交叉注意(SDPA)`，全部条件支路**零初始化**（初始恒等，可无损续训）。
- 已在 **353 说话人 / 29.1 万条**数据上从零训练；跑到 **step 170,000（epoch 55，~13.5h）**，loss ≈ 0.4；本地 zero-shot 推理验证通过（与参考音色相似度 0.777 → **0.924**）。
- 长音频已用 **fish-audio-preprocess** 切成 ≤10s；预处理产物 **fp16** 存储并支持**断点续跑**。

---

## 1. 目标与方案

### 1.1 目标
基础模型（DDSP-SVC 6.3）原本是**单说话人/离散查表**音色：`Unit2Control` 里 `nn.Embedding(n_spk, dim)`，`n_spk=1` 时根本没有音色条件。目标是改为**连续说话人条件**，实现"参考音频 → 音色向量 → 注入模型"的 zero-shot。

### 1.2 方案（C 档，最终采用）
| 机制 | 实现 | 作用 |
|---|---|---|
| 全局说话人向量 `spk_vec` | WavLM-SV (`microsoft/wavlm-base-plus-sv`) x-vector，512 维，L2 归一化 | 粗粒度音色 |
| 加性注入 + 逐层 FiLM | `spk_proj`（零初始化）+ `ConformerNaiveEncoder.films`（每层 `x*(1+γ)+β`，零初始化） | DDSP 控制网络音色调制 |
| reflow 侧 FiLM | `LYNXNet2.spk_films`（每 block，零初始化） | 频谱细化侧音色调制 |
| 说话人 token 交叉注意 | 参考音频 WavLM 帧特征 → 自适应池化到 K=16 token → `CrossAttention`(SDPA) 注入 `Unit2Control`（零初始化 out_proj） | 细粒度/在上下文音色 |
| 防泄漏 | 训练时参考取**同说话人另一句**（cross-utterance）+ 噪声 + dropout | 防止模型抄参考内容 |
| 多参考 | 推理 `-r a.wav,b.wav`：向量取平均、token 拼接 | 提升鲁棒性 |

**关键性质**：所有新增支路零初始化 → 初始为恒等映射（实测 identity-at-init `max diff = 0.0`），因此可**无损加载旧基座并续训**。

### 1.3 参数规模
- 原基座：**54.9M**。
- 开启全部 spk 条件（512/6 配置）：**72.9M**（+18M，主要来自 FiLM 线性层）。
- 本次正式训练用的是更大的 DDSP 骨干（`n_aux_layers:8`、`n_aux_chans:768`），参数更大。

---

## 2. 运行环境

### 2.1 本地开发机（小显存验证）
- Conda env `310`，Python 3.10.21
- 解释器：`C:\Users\zibo9\anaconda3\envs\310\python.exe`
- torch 2.14.0+cu130、torchaudio 2.11.0、transformers 5.17.0、librosa 0.11.0、**numpy 1.26.4（锁定）**、scipy、onnx/onnxruntime、soundfile/sounddevice、tensorboard、matplotlib、FreeSimpleGUI
- GPU：**RTX 3050 Laptop 4GB**

### 2.2 远程训练机（AutoDL）
- 主机：`AutoDL 实例（地址见本地记录）`，用户 `root`，项目在 `/root/DDSP-SVC-6.3`（**凭据不入库**）
- 解释器：`/root/miniconda3/bin/python`（**base**，Python 3.10.8）
- torch **2.5.1+cu121**、transformers 4.52.4、numpy 1.26.4
- GPU：**RTX 4080 SUPER 32GB**；**128 核**；**503GB RAM**
- 磁盘：`/` 30G（系统盘）、`/root/autodl-tmp` 818G（数据盘）、`/root/autodl-fs` 200G（公共/持久存储）
- 免密 SSH：本地私钥 `C:\Users\zibo9\AppData\Local\Temp\opencode\ddsp_ssh_key`

> **Windows→ssh 双引号会被剥离**：远程命令一律写成脚本文件再执行（`bash /root/xxx.sh`），或本机用 paramiko 助手 `rexec.py`。

---

## 3. 数据

### 3.1 原始数据（远端）
- 原始素材池：`/root/autodl-tmp/dataset/11/<说话人>/*.wav`（**353 说话人 / 264,016 文件 / 151G**）
- 预处理输入：`/root/autodl-tmp/ddsp-data/train/audio/`（与原始池**硬链接**，不额外占盘）；val 在 `.../val/audio/`
- 目录约定：`audio/<说话人id>/*.wav`。**子目录名 = 说话人分组键**（用于跨句配对/留出评估），连续向量方案下 `n_spk` 保持 1。

### 3.2 长音频切片（fish-audio-preprocess）
- 原因：>10s 的长音频会让预处理 worker CUDA OOM。
- 处理：>10.5s 的文件用 `fish_audio_preprocess.utils.slice_audio.slice_audio_file`（静音感知，min_duration=6, max_duration=10）切为 ≤10s 片段；原文件从 `ddsp-data/*/audio` 删除（原始池为硬链接，保留），并清理其旧产物。
- 结果：train wav **264,016 → 291,695**（22,057 个长文件 → 49,736 段）；val 6 → 18；无文件 >10.5s。
- 安装：`pip install --no-deps fish-audio-preprocess`（**不污染 numpy/torch**；切片只用 librosa/numpy/soundfile）。
- 切片脚本：`/root/reslice.py`（16 进程）。

### 3.3 预处理产物（每个 split 下）
```
units/<spk>/x.wav.npy      float16 [T,768]   ContentVec 768L12-TTA2X
f0/    volume/  aug_vol/    float32           RMVPE 音高 / 音量包络
mel/  aug_mel/             float16 [T,128]   目标 log-mel / 增广
spk/<spk>/x.wav.npy        float32 [512]     WavLM-SV 全局向量
spk_tokens/<spk>/*.npy     float16 [16,768]  说话人 token
pitch_aug_dict.npy         增广 keyshift 字典（{相对路径: keyshift}）
skip/                      F0 提取失败被移出的文件（train 709 个）
```
- **fp16 存储**：units/mel/aug_mel（训练本就 fp16/bf16 autocast，无损，省一半磁盘）。总量 ~391G（fp32 会到 770G，写满 818G 盘）。
- 最终规模：train 音频 **290,986**、产物 **290,986 齐全**；val **18/18**；`complete w/o aug entry = 0`。

### 3.4 预处理命令
```sh
# 主特征 + spk/token（新数据一条搞定）
cd /root/DDSP-SVC-6.3
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  setsid /root/miniconda3/bin/python -u preprocess.py \
  -c configs/reflow_multispk.yaml -j 10 -d cuda --max-duration 15 \
  > /root/preprocess.log 2>&1 < /dev/null &

# 只补 spk/token（已有 units/mel 时）
/root/miniconda3/bin/python preprocess_spk.py -c configs/reflow_multispk.yaml -d cuda
```

### 3.5 `preprocess.py` 的两个重要特性
1. **断点续跑**：已存在"全部产物 + pitch_aug 条目"的文件自动跳过；`--force` 全量重算。
2. **pitch_aug_dict 与产物对齐**：续跑判断要求 dict 有条目，否则会重新处理（自愈）；dict 每 2000 个完成落盘一次（崩溃最多丢 2000 条，下轮自动补）。
   - **原因**：`data_loaders.get_data` 在增广分支读 `self.pitch_aug_dict[name_ext]`，缺条目会训练时 `KeyError`。
   - 收尾校验：`/root/verify_align2.py`，期望 `complete w/o aug ent = 0`。

---

## 4. 代码改动清单

### 4.1 新增
| 文件 | 说明 |
|---|---|
| `encoder/spk_encoder.py` | WavLM-SV 封装：`encode()`(512)、`encode_tokens()`([K,768])、`encode_files()`、`encode_tokens_files()`；含 torch<2.6 回退补丁 `_from_pretrained_with_fallback` |
| `preprocess_spk.py` | 独立、可续跑的 spk/token 提取脚本 |

### 4.2 修改（核心）
| 文件 | 改动 |
|---|---|
| `ddsp/unit2control.py` | `spk_proj`+`spk_norm`（加性）、逐层 FiLM、`CrossAttention`(SDPA，零初始化 out_proj) |
| `ddsp/model_conformer_naive.py` | `ConformerNaiveEncoder` 支持 `film_cond_dim`（逐层 FiLM） |
| `ddsp/vocoder.py` | `CombSubSuperFast` 透传 `spk_vec`/`spk_tokens` |
| `reflow/lynxnet2.py` | `LYNXNet2` 加 `spk_dim` + `spk_films`（逐 block FiLM） |
| `reflow/reflow.py` | `RectifiedFlow` 透传 `spk_cond`（loss/euler/rk4 全部） |
| `reflow/vocoder.py` | `Unit2Wav` 新参数、`load_model_vocoder`（`strict=False`） |
| `reflow/data_loaders.py` | 加载 `spk_vec`/`spk_tokens`；跨句配对；噪声/dropout；大数据 mmap 懒加载 |
| `reflow/solver.py` | train/test 传 `spk_vec`/`spk_tokens` |
| `train_reflow.py` | 传 spk 架构参数 |
| `main_reflow.py` / `batch_infer.py` / `gui_reflow.py` | 新增 `-r/--reference`（多参考） |
| `preprocess.py` | 集成 spk/token；units/mel/aug_mel 存 fp16；断点续跑；`--max-duration` |
| `configs/reflow.yaml` | zero-shot 配置 + `batch_size:32` |
| `logger/utils.py` | DotDict 修复（见踩坑） |
| `configs/reflow_multispk.yaml` | 目标机多说话人训练配置（新增） |

### 4.3 早期 6.3 精简（前一阶段）
- `ddsp/vocoder.py` 仅保留 `rmvpe` + `contentvec768l12tta2x`；`ddsp/core.py` 仅留 `upsample`；删 `ddsp/loss.py`。
- `requirements.txt` 去掉 parselmouth/pyworld/resampy/torchcrepe/torchfcpe。
- GUI f0 列表仅 `rmvpe`。

---

## 5. 配置

### 5.1 `configs/reflow.yaml`（本地/单说话人验证）
关键：`use_spk/use_spk_film/use_spk_reflow/use_spk_tokens: true`，`spk_dim:512`，`spk_token_dim:768`，`spk_n_tokens:16`，`batch_size:32`。

### 5.2 `configs/reflow_multispk.yaml`（目标机正式训练）
```yaml
data:
  train_path: /root/autodl-tmp/ddsp-data/train
  valid_path: /root/autodl-tmp/ddsp-data/val
model:
  n_aux_layers: 8      # 加宽 DDSP 骨干
  n_aux_chans: 768
  n_layers: 6
  n_chans: 1024
  use_spk: true
  spk_dim: 512
  use_spk_film: true
  use_spk_reflow: true
  use_spk_tokens: true
  spk_token_dim: 768
  spk_n_tokens: 16
  spk_backend: 'wavlm-sv'
  spk_model: 'microsoft/wavlm-base-plus-sv'
env:
  expdir: /root/autodl-tmp/ddsp-multispk   # 数据盘
train:
  num_workers: 12
  amp_dtype: bf16
  batch_size: 96
  cache_all_data: false     # 29万文件不能塞 RAM
  cache_fp16: true
  decay_step: 30000         # 每 ~10 epoch ×0.9（原来 4000 太快）
  gamma: 0.9
  lr: 0.0005
  interval_val: 2000
  interval_force_save: 10000
  spk_cross_prob: 0.5
  spk_noise: 0.02
  spk_drop_prob: 0.1
```

---

## 6. 训练

### 6.1 从零训练（推荐，大数据 + 大 GPU）
```sh
cd /root/DDSP-SVC-6.3
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  setsid /root/miniconda3/bin/python -u train_reflow.py -c configs/reflow_multispk.yaml \
  > /root/autodl-tmp/train.log 2>&1 < /dev/null &
```

### 6.2 从已有基座 FT
- 条件支路零初始化 → 初始恒等，可无损续训。
- **架构超参必须与基座一致**，否则 `strict=False` 会**静默丢权重**（`n_aux_chans/n_aux_layers/n_chans/n_layers/win_length/block_size/encoder`）。
- 不要复用旧 expdir（保存/删除逻辑可能覆盖旧 ckpt）；把基座 ckpt 拷进新 expdir 即可（`load_model` 自动取最大 step）。

### 6.3 实测
- ~**3.8 step/s**（每 epoch 3032 步 ≈ 13 min）；显存仅 **~8.7GB / 32GB**（batch 可提到 192，未试）。
- 跑到 **step 170,000（epoch 55，13.5h）**，`train/reflow_loss` 稳定下降，loss ≈ 0.4，lr 按 `decay_step=30000` 衰减到 **2.95e-4**。
- 4GB 本地机 batch 悬崖：B=16→1.23、**B=32→2.38**、B=48→0.21 step/s（务必用 32）。

### 6.4 监控与 TensorBoard
```python
!sh /root/monitor_train.sh          # 步骤/loss/显存
!tail -f /root/autodl-tmp/train.log
```
- AutoDL 自带 TensorBoard：`tensorboard --logdir /root/tf-logs --port 6007`。
- 软链训练日志进去即可在实例面板「TensorBoard」查看：
  ```sh
  ln -sfn /root/autodl-tmp/ddsp-multispk/logs /root/tf-logs/ddsp-multispk
  curl -s http://127.0.0.1:6007/data/runs     # -> ["ddsp-multispk"]
  ```

---

## 7. 推理（zero-shot）

```powershell
# 单条（本地 4GB 实测可用）
& "C:\Users\zibo9\anaconda3\envs\310\python.exe" main_reflow.py `
  -m exp/ddsp-multispk/model_100000.pt `
  -i dry.wav -o out.wav `
  -r ref1.wav,ref2.wav
# 批量
& "..." batch_infer.py -m <ckpt> -i <indir> -o <outdir> -r ref.wav
# GUI（实时）：模型 + 参考音频栏
& "..." gui_reflow.py
```
- **必须用 ckpt 同目录的 `config.yaml`**（`load_model_vocoder` 会读取；缺了报错）。
- 不传 `-r` 则用零向量（不是训练集均值，效果随机，仅兜底）。
- 常用参数：`-k` 变调(半音)、`-f` 共振峰、`-step` 采样步数、`-method euler/rk4`、`-ts` t_start。
- 中文名文件在 PowerShell 传参可能乱码；用 Python `glob` + `subprocess`（list 传参）最稳。

### 7.1 本地 zero-shot 验证结果（model_100000）
- 输入：`霜雪千年（干声）.wav`（44.1k 立体声 235s）；参考：`zh_vo_Main_Linaxita_2_4_57_14.wav`（12s）
- 输出：`exp/ddsp-multispk/zs_out.wav`（235.1s，峰值 0.933，RMS 0.232），耗时 **116.7s（~0.5× 实时）**
- `cos(out, ref) = 0.9237` vs `cos(in, ref) = 0.7768` → 音色明显向参考靠拢 ✅
- 备注：用的是训练同一 WavLM-SV，属自证；严格评估需独立 SV 模型 + held-out 说话人。

---

## 8. 关键踩坑汇总

1. **Python 3.10 + spawn**：`DotDict.__getattr__` 对 dunder 返回 `None` 导致 pickle 崩（`TypeError: 'NoneType' object is not callable`）。已修为 dunder 抛 `AttributeError`（`logger/utils.py`、`ddsp/vocoder.py`、`reflow/vocoder.py`）。
2. **torch < 2.6 加载 WavLM-SV**：transformers≥4.51 因 CVE-2025-32434 拒绝 `torch.load` 的 `.bin`。已在 `encoder/spk_encoder.py` 加自动回退；**首选升级 torch≥2.6**。目标机连不上 HF，必须 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`（否则每次 from_pretrained 重试 ~400s）。
3. **多进程线程超订阅**：128 核下每个 torch 进程默认开满线程 → 预处理从 ~20 文件/s 掉到 <0.1。必须 `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`。
4. **显存悬崖**：预处理 `-j 12` → 32.2/32.7GB（险）；`-j 10` 稳。训练 4GB 上 batch 必须 32。
5. **`nn.MultiheadAttention` 在 fp16/4GB 卡死**：换 `F.scaled_dot_product_attention` 版 `CrossAttention`。
6. **长音频 OOM**：>10s 直接预处理会爆；用 fish 切片到 ≤10s，并保留 `--max-duration` 兜底截断。
7. **磁盘爆**：fp32 units 会占 ~570G（总量 770G > 818G）→ 改 fp16（~391G）。
8. **对齐隐患**：产物齐全但 `pitch_aug_dict` 缺条目 → 训练 `KeyError`。续跑判断要求 dict 条目 + 定期落盘。
9. **ckpt 加载**：`load_model_vocoder` / `load_model` 均 `strict=False`；旧 ckpt 无 spk 键自动退化为无 spk 模型。
10. **多进程重启前清理孤儿**：`pkill -9 -f preprocess.py; pkill -9 -f spawn_main`（Windows `pkill -f` 会误杀执行命令的 shell，远程用 `pkill`/显式 PID，注意不要匹配自身）。

---

## 9. 当前进度 / 状态

| 项 | 状态 |
|---|---|
| 代码改造 | ✅ 完成（本地 + 远端同步） |
| 预处理 | ✅ 完成：train 290,986、val 18、`complete w/o aug ent = 0` |
| 训练 | ⏹ 已在 step 170,000 终止（epoch 55，loss≈0.4） |
| 训练结果 | 17 个 ckpt（10000…170000）+ config + log_info + logs + train.log |
| 本地已下载 | `exp/ddsp-multispk/model_100000.pt` + `config.yaml`（MD5 校验一致） |
| zero-shot 验证 | ✅ 本地推理通过，音色相似度 0.777→0.924 |

### 9.1 结果位置
- 远端公共存储（持久，已复制）：
  - `/root/autodl-fs/DDSP-SVC-6.3`（767M，完整工程）
  - `/root/autodl-fs/ddsp-multispk`（9.1G，训练结果）
- 远端数据盘（关机不丢，释放实例才丢）：`/root/autodl-tmp/ddsp-data`、`/root/autodl-tmp/ddsp-multispk`
- 本地：`D:\DDSP-SVC-6.3`（源码 + `exp/ddsp-multispk/model_100000.pt`）

### 9.2 远端辅助脚本
`/root/run_pre.sh`（预处理）、`/root/measure.sh`、`/root/measure2.sh`、`/root/kill_pre.sh`、`/root/reset.sh`、`/root/reslice.py`、`/root/verify_align2.py`、`/root/run_train.sh`、`/root/monitor_train.sh`、`/root/copy_to_fs.sh`。
（安装的工具：`nvitop`、`btop`、`fish-audio-preprocess`）

---

## 10. 待办 / 下一步

1. **`export_onnx.py` 未同步**：其 `spk_mix` 输入仍是旧设计，需改为 `spk_vec`/`spk_tokens` 才能导出带条件的 ONNX。
2. **继续训练**：从 `model_170000.pt` 续训（`load_model` 自动取最大 step），可把 `batch_size` 提到 192 提速；若显存允许可加宽 reflow。
3. **严格评估**：留出**未参与训练的说话人**，用**独立 SV 模型**（非训练用 WavLM-SV）算 **SECS**；再算内容保持（ASR/WER）。
4. **提升音色**：换更强说话人编码器（ERes2NetV2/ECAPA/多编码器集成）；多参考；或 few-shot 微调 `spk_proj`。
5. **val 说话人与 train 重叠**：当前 val（丸子/阿拉丁）与 train 重叠，只能当损失监控，不能评估 zero-shot 泛化。

---

## 11. 快速命令速查

```powershell
# 本地编译自检
& "C:\Users\zibo9\anaconda3\envs\310\python.exe" -m py_compile ddsp/unit2control.py reflow/vocoder.py encoder/spk_encoder.py

# 本地 zero-shot 推理
& "C:\Users\zibo9\anaconda3\envs\310\python.exe" main_reflow.py `
  -m exp/ddsp-multispk/model_100000.pt -i dry.wav -o out.wav -r ref.wav
```
```sh
# 远端：预处理 / 续跑
sh /root/run_pre.sh
# 远端：训练
sh /root/run_train.sh
# 远端：监控
sh /root/monitor_train.sh
# 远端：对齐校验
/root/miniconda3/bin/python /root/verify_align2.py
# 远端：TensorBoard 软链
ln -sfn /root/autodl-tmp/ddsp-multispk/logs /root/tf-logs/ddsp-multispk
```

---

## 12. 安全

- 远端 SSH 密码**不要**写入本仓库/README/notebook；建议定期更换。
- 本地免密私钥在 `C:\Users\zibo9\AppData\Local\Temp\opencode\ddsp_ssh_key`（临时目录），可随时删除以撤销访问。
