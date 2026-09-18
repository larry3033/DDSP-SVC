# AGENTS.md — DDSP-SVC 6.3 操作指南（含 Zero-shot 改造）

> 本文件是项目记忆/操作手册。后续会话请优先阅读本文件，再动代码。
> **完整交接文档见 `README_ZEROSHOT.md`**（方案、改动、数据、训练、推理、踩坑、进度、待办）。
> 项目根：`D:\DDSP-SVC-6.3`（**非 git 仓库**，删改不可恢复，务必先备份）。
> 历史改动备份：`C:\Users\zibo9\AppData\Local\Temp\opencode\ddsp63_backup`

---

## 1. 运行环境（固定）

- Conda env：`310`（Python 3.10.21）
- 解释器：`C:\Users\zibo9\anaconda3\envs\310\python.exe`
- 关键依赖：torch 2.14.0+cu130、torchaudio 2.11.0、transformers 5.17.0、librosa 0.11.0、numpy **1.26.4（锁定）**、scipy、onnx/onnxruntime、soundfile/sounddevice、tensorboard、matplotlib、FreeSimpleGUI
- GPU：RTX 3050 Laptop **4 GB**（driver CUDA 13.3）
- 联网可用（HF / hf-mirror / ModelScope / PyPI）
- 调 Python 一律用上面的绝对路径，**不要用裸 `python`**（系统里还有另一套 anaconda base，勿混用）

---

## 2. 预训练资源

| 用途 | 路径 |
|---|---|
| 内容编码 ContentVec 768L12-TTA2X | `pretrain/contentvec/pytorch_model.bin` |
| 音高 RMVPE | `pretrain/rmvpe/model.pt` |
| 声码器 NSF-HiFiGAN | `pretrain/nsf_hifigan/model`（`pc_aug: true`） |
| 说话人编码 WavLM-SV | HF 缓存 `microsoft/wavlm-base-plus-sv`（首次自动下载） |

---

## 3. 数据布局与预处理

```
data/train/audio/<说话人id>/*.wav
data/val/audio/<说话人id>/*.wav
```
预处理产出（每个目录下）：
- `units/` `f0/` `volume/` `mel/` `aug_mel/` `aug_vol/` `pitch_aug_dict.npy`
  - **`units`/`mel`/`aug_mel` 存 float16**（省一半磁盘；训练本来就是 fp16/bf16 autocast，无损）；`f0`/`volume`/`aug_vol` 仍 fp32
- `spk/<name>.wav.npy`（512-d WavLM x-vector，`use_spk` 时）
- `spk_tokens/<name>.wav.npy`（`[K,768]` fp16，`use_spk_tokens` 时）

命令：
```powershell
& "C:\Users\zibo9\anaconda3\envs\310\python.exe" preprocess.py     -c configs/reflow.yaml -j 2
& "C:\Users\zibo9\anaconda3\envs\310\python.exe" preprocess_spk.py -c configs/reflow.yaml
```
- **`preprocess.py` 支持断点续跑**：已存在全部产物（units/f0/volume/mel/aug_mel/aug_vol[/spk][/spk_tokens]）的文件自动跳过；`pitch_aug_dict.npy` 会与旧值合并（续跑不丢增广 keyshift）。`--force` 可强制全量重算。
- `preprocess_spk.py` 也可续跑：缺 spk 补 spk、缺 token 补 token，已存在的跳过；`--force` 全量重算。
- 已有数据若只补 token，直接跑 `preprocess_spk.py` 即可（不会重算 units/mel）。

---

## 4. Zero-shot 配置（`configs/reflow.yaml`）

```yaml
model:
  n_spk: 1                 # 连续向量方案：保持 1，不建查表
  use_spk: true            # 开总开关
  spk_dim: 512             # WavLM-SV x-vector 维度
  use_spk_film: true       # DDSP 控制网络逐层 FiLM
  use_spk_reflow: true     # reflow(LYNXNet2) 逐 block FiLM
  use_spk_tokens: true     # 参考 token 交叉注意
  spk_token_dim: 768
  spk_n_tokens: 16         # 显存紧张可降到 8
  spk_backend: 'wavlm-sv'
  spk_model: 'microsoft/wavlm-base-plus-sv'
train:
  batch_size: 32           # ⚠️ 4GB 卡必须 32；48 会显存抖动到 0.2 step/s
  cache_all_data: true     # 大数据集改 false（否则爆内存）
  spk_cross_prob: 0.5      # 跨句配对概率（防内容泄漏）
  spk_noise: 0.02
  spk_drop_prob: 0.1
```

---

## 5. 训练

### 5.1 从零训练（大数据 + 大 GPU 推荐）
新 expdir，直接：
```powershell
& "C:\Users\zibo9\anaconda3\envs\310\python.exe" train_reflow.py -c <新config>
```

### 5.2 从现有基座 FT（当前 4GB 路线）
`model_2000.pt` 是**单说话人 furina** 基座。FT 步骤：
```powershell
New-Item -ItemType Directory -Force -Path exp\reflow-multispk
Copy-Item exp\reflow-test\model_2000.pt exp\reflow-multispk\model_2000.pt
# 新 config 设 env.expdir: exp/reflow-multispk，再 train_reflow.py -c <新config>
```
- **不要复用旧 expdir**（保存/删除逻辑可能覆盖 `model_2000.pt`）。
- 条件支路零初始化 → 初始恒等，FT 起始行为与旧模型完全一致，稳定。

### 5.3 约束
- **架构超参必须与基座一致**：`n_aux_chans:512`、`n_chans:1024`、`n_layers:6`、`win_length:2048`、`block_size:512`、`encoder`、`encoder_out_channels`。不一致时 `strict=False` 会**静默丢权重**。
- 优化器状态不保存（`save_opt:false`）：优化器重启，LR 按 `global_step` 续衰减，正常。

---

## 6. 推理

```powershell
# 单条，零样本
& "C:\Users\zibo9\anaconda3\envs\310\python.exe" main_reflow.py `
  -m exp/<expdir>/model_XXXX.pt -i input.wav -o out.wav `
  -r ref1.wav,ref2.wav        # 多参考：向量取平均、token 拼接；不传则用零向量
# 批量
& "C:\Users\zibo9\anaconda3\envs\310\python.exe" batch_infer.py -m <ckpt> -i <indir> -o <outdir> -r ref.wav
# GUI（实时）：模型/参考音频栏选择，其余同原版
& "C:\Users\zibo9\anaconda3\envs\310\python.exe" gui_reflow.py
```
- 常见参数：`-k` 变调、`-f` 共振峰、`-step` 采样步数、`-method euler/rk4`、`-ts` t_start。

---

## 7. 关键踩坑（务必记住）

1. **Python 3.10 + multiprocessing spawn**：`DotDict.__getattr__` 对 dunder 返回 `None` 会崩（`TypeError: 'NoneType' object is not callable`）。已修为对 dunder 抛 `AttributeError`（改在 `logger/utils.py`、`ddsp/vocoder.py`、`reflow/vocoder.py`）。
2. **batch_size=48 显存悬崖**：4GB 下实测 B=16→1.23、**B=32→2.38 step/s**、B=48→0.21 step/s（抖动）。默认 32。
3. **交叉注意用 SDPA**：`nn.MultiheadAttention` 在 fp16/4GB 下会卡死，已换 `F.scaled_dot_product_attention` 版 `CrossAttention`（`ddsp/unit2control.py`）。
4. **ckpt 加载**：`reflow/vocoder.py:load_model_vocoder` 与 `logger/utils.py:load_model` 均 `strict=False`；旧 ckpt 无 spk 键时自动退化为无 spk 模型。
5. **零初始化**：`spk_proj`、FiLM、`CrossAttention.out_proj` 全零初始化 → 初始恒等（identity-at-init diff=0.0），可无损续训。
6. **跨句配对**：训练时参考必须取**同说话人另一句**，避免模型抄参考内容；已实现（`data_loaders.py`）。
7. **大数据内存**：`cache_all_data: true` 会把全部 mel/units 塞进 RAM；大数据务必改 `false`。
8. **别混用另一套 anaconda base python**（有 `main.py` 在跑）。
9. **torch < 2.6 加载 WavLM-SV 报错**：transformers≥4.51 因 CVE-2025-32434 拒绝 `torch.load` 的 `.bin`，而 `microsoft/wavlm-base-plus-sv` 只有 `.bin`。已在 `encoder/spk_encoder.py` 加自动回退（`_from_pretrained_with_fallback`，捕获该 ValueError 后打补丁重试）。**首选方案仍是升级 torch ≥ 2.6**。目标机（AutoDL）默认落在 `/root/miniconda3` base 环境，注意换到装了依赖的环境。

---

## 8. 文件地图（zero-shot 相关）

**新增**
- `encoder/spk_encoder.py` — WavLM-SV：`encode()` / `encode_tokens()` / `encode_files()` / `encode_tokens_files()`
- `preprocess_spk.py` — 独立、可续跑 spk/token 提取

**核心改动**
- `ddsp/unit2control.py` — `spk_proj`+`spk_norm`、`CrossAttention`(SDPA)、逐层 FiLM
- `ddsp/model_conformer_naive.py` — `ConformerNaiveEncoder` 支持逐层 FiLM（`film_cond_dim`）
- `ddsp/vocoder.py` — `CombSubSuperFast` 透传 `spk_vec`/`spk_tokens`
- `reflow/lynxnet2.py` — `LYNXNet2` 加 `spk_films`
- `reflow/reflow.py` — `RectifiedFlow` 透传 `spk_cond`
- `reflow/vocoder.py` — `Unit2Wav` 新参数、`load_model_vocoder`
- `reflow/data_loaders.py` — 加载 spk_vec/spk_tokens、跨句配对、噪声/dropout
- `reflow/solver.py`、`train_reflow.py` — 训练/验证传条件
- `main_reflow.py`、`batch_infer.py`、`gui_reflow.py` — `-r/--reference`
- `preprocess.py` — 集成 spk/token 提取
- `configs/reflow.yaml` — zero-shot 配置

**参数规模**：54.9M → **72.9M**（+18M，主要来自 FiLM 线性层）

---

## 9. 快速自检

```powershell
# 编译
& "C:\Users\zibo9\anaconda3\envs\310\python.exe" -m py_compile ddsp/unit2control.py reflow/vocoder.py encoder/spk_encoder.py
# 旧基座兼容性
& "C:\Users\zibo9\anaconda3\envs\310\python.exe" main_reflow.py -m exp/reflow-test/model_2000.pt -i data/val/audio/1/vo_FDAQ006_9_furina_03.wav -o out.wav
```
- 期望：模型 `72.9M` 参数（开全部 spk 时）；identity-at-init diff = 0.0。

---

## 10. 未完成 / 注意事项

- **`export_onnx.py` 未同步**：其 `spk_mix` 输入需改为 `spk_vec`/`spk_tokens` 才能导出带条件的 ONNX。
- 评估请**留出未参与训练的说话人**，用独立 SV 模型算 **SECS**，别用训练用的 WavLM 自证。
- 测试产物：`exp/reflow-zs-test/`（从零 200 步）、`exp/reflow-test/`（原 furina `model_2000.pt`），可删。

---

## 11. 目标机（AutoDL RTX 4080 SUPER 32GB）运行要点

- 主机：`AutoDL 实例（地址见本地记录）`，用户 `root`，容器内项目 `/root/DDSP-SVC-6.3`（**凭据不入库**）。
- 环境：`/root/miniconda3/bin/python`（**base**，Python 3.10.8）、torch **2.5.1+cu121**、transformers 4.52.4、numpy 1.26.4、128 核、503GB RAM、`/root/autodl-tmp` 818G（数据盘）。
- 数据：`/root/autodl-tmp/ddsp-data/{train,val}`；train = **353 说话人 / 264017 文件**，val = 6 文件（2 说话人，**与 train 重叠，不能用于 zero-shot 评估**，需另留 held-out 说话人）。

### 必须设置的环境变量
```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1        # 目标机连不上 HF，否则 from_pretrained 重试 ~400s/进程
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```
- **线程上限是关键**：128 核下多进程 torch 默认每进程开满线程 → 严重超订阅，预处理会从 ~20 文件/s 掉到 <0.1 文件/s。加上述变量后恢复。

### 预处理命令（可直接用 nohup 后台）
```sh
cd /root/DDSP-SVC-6.3
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  setsid /root/miniconda3/bin/python -u preprocess.py -c configs/reflow_multispk.yaml -j 10 -d cuda --max-duration 15 \
  > /root/preprocess.log 2>&1 < /dev/null &
```
- 实测：`-j 12` → 32210/32760 MiB（太险）；**`-j 10` → ~22536 MiB、~25 文件/s**，约 3h。
- **长音频已用 `fish-audio-preprocess` 切片**：>10.5s 的文件切成 ≤10s 片段（`fish_audio_preprocess.utils.slice_audio.slice_audio_file`，静音感知；`pip install --no-deps fish-audio-preprocess` 不污染 numpy/torch）。切片后 train wav 由 264,016 → **291,695**，val 6 → 18。原文件从 `ddsp-data/*/audio` 删除（原始池 `dataset/11` 为硬链接，保留）。切片脚本 `/root/reslice.py`（16 进程）。
- `preprocess.py` 的 `--max-duration`（默认 30，本次用 15）是兜底截断，切片后已无文件超 10.5s。
- **磁盘**：264k 文件时 fp32 units 会占 ~570G（总量 ~770G）→ 会写满 818G。已改 **units/mel/aug_mel 存 fp16**（总量 ~391G，峰值加原始音频 151G ≈ 542G，余 ~276G）。
- **`preprocess.py` 支持断点续跑**（见 §3），中断后直接重跑即可；`--force` 全量重算。
- 重启前建议 `pkill -9 -f preprocess.py; pkill -9 -f spawn_main`（否则残留孤儿 worker 占满显存）。
- 远程辅助脚本：`/root/run_pre.sh`（启动）、`/root/measure.sh`、`/root/measure2.sh`（含 dtype 校验）、`/root/kill_pre.sh`（清理）、`/root/reset.sh`（清空产物）、`/root/reslice.py`（长音频切片）、`/root/verify_align2.py`（对齐校验）。
- **对齐校验**：`data_loaders.get_data` 在增广分支用 `self.pitch_aug_dict[name_ext]`。续跑判断已要求"产物齐全 **且** dict 有条目"，否则会 `KeyError`；`pitch_aug_dict` 每 2000 个完成落盘一次。收尾务必跑 `verify_align2.py`，期望 `complete w/o aug ent = 0`（train 290,986 / val 18，skip 709）。
- 远程调试用 Windows OpenSSH 时**双引号会被剥离**，命令请写成脚本文件再 `bash /root/xxx.sh` 执行（或改用 paramiko 助手直传）。

### 训练（目标机 multispk）
```sh
cd /root/DDSP-SVC-6.3
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 setsid /root/miniconda3/bin/python -u train_reflow.py -c configs/reflow_multispk.yaml \
  > /root/autodl-tmp/train.log 2>&1 < /dev/null &
```
- `configs/reflow_multispk.yaml`：`expdir=/root/autodl-tmp/ddsp-multispk`（**数据盘**）、`cache_all_data:false`、`num_workers:12`、`batch_size:96`、`bf16`、`n_aux_layers:8`/`n_aux_chans:768`（从零训练）。
- 实测：**~3.8 step/s**、显存仅 **8.7GB/32GB**（可把 batch 提到 192）。
- 启动脚本 `/root/run_train.sh`，监控 `/root/monitor_train.sh`。
- **TensorBoard（AutoDL 实例监控）**：平台自带 `tensorboard --logdir /root/tf-logs --port 6007`。把训练日志软链进去即可在 AutoDL 面板的「TensorBoard」看到：
  ```sh
  ln -sfn /root/autodl-tmp/ddsp-multispk/logs /root/tf-logs/ddsp-multispk
  ```
  验证 `curl -s http://127.0.0.1:6007/data/runs` 输出 `["ddsp-multispk"]`。
