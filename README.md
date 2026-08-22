# H3VM Multi-GPU Loader for ComfyUI

MiniMax H3 多卡加载与显存调度节点。目标不是提供一条固定工作流，而是把 H3 的模型加载、Turbo LoRA、普通 Style/角色 LoRA、步数、时长、分辨率、GPU 模式和 Video VAE 调度收进一个统一入口，方便接入任意 H3 工作流。

> 当前版本：v0.20.0-rc5

## 两种使用形态

### 1. H3VM Multi-GPU Loader｜H3多卡加载器

独立使用的一体化主控。继续负责模型、Turbo、Steps、Prompt、分辨率、Seed 等便捷生成控制，适合直接生成和快速 A/B。

### 2. H3VM Core｜多卡执行引擎

用于接入复杂主工作流，接口边界是 `MODEL -> H3VM execution -> MODEL`。主工作流先完成 Ref2VA/FL2VA、Turbo、普通 LoRA、Sigma/Scheduler 等业务路由，再把最终 MODEL 交给 Core。Core 只负责 GPU 模式、显存/执行调度和 telemetry，不修改 Prompt、Seed、分辨率、Sampler、Sigmas 或实际采样步数。

当前 Generic Core 已支持 clean H3 和普通 ModelPatcher weight patches（常见 `LoraLoaderModelOnly`）。runtime injections、object/hook/weight-wrapper patches 仍 fail-closed，防止 GPU0/GPU1 计算不同模型。

## 作者 / Author

- GitHub: `bt420000-collab`
- Bilibili: https://space.bilibili.com/16826253


## 核心节点

`H3VM Multi-GPU Loader｜H3多卡加载器`

常用设置都集中在一个节点中：

- H3 diffusion model
- Turbo LoRA 与强度
- 4 / 6 / 8 步
- 时长（秒，可自由填写）
- 画面比例与常用分辨率预设
- Seed
- GPU 运行模式
- 双卡扩显存实验档位
- Prompt（放在最下方，默认折叠，展开后为大输入框）

## GPU 模式

### 单卡｜标准模式

只使用主 GPU。适合作为质量、性能和显存基线。

### 双卡协同｜日常推荐

保留当前成熟的双卡协同计算与 Dual Video VAE。主要价值是分担负载、降低单卡持续满载和风扇压力；在当前测试机上也有小幅性能收益。

### 双卡扩显存｜高清模式

以“单卡 OOM 时仍能完成推理”为第一目标。通过 RAM backing、QKV head shard、MLP token shard、micro-chunk 和动态显存驻留降低单卡峰值显存。

当前实测建议使用 **8 步**。在当前 16GB + 8GB 测试机上，4 步 Capacity 曾出现明显音质下降；INT8 Attention 与官方 optimized/Sage Attention A/B 音质一致，因此当前不把问题归因于 Attention kernel。

### 双卡满血｜原版20步

模式 4 现已接入历史 Dev9.4 Stock H3 FullThrottle 路径。该模式忽略 Turbo LoRA 与 4/6/8 步预设，固定使用原版 H3 `res_multistep` + 20 steps，不应用 Turbo SigmaShift；普通 Style / 角色 LoRA Stack 仍然生效。双卡执行使用 22/28 Snapshot Islands、2GB RAM backing 与 linear predictor 0.75。

注意：这里的“原版”指 **原版 H3 权重/采样契约**；Dev9.4 FullThrottle 执行器本身使用 stale/predicted boundary snapshot，是实验性近似双卡推理，并非逐层严格同步的 exact execution。

## 双卡扩显存档位

该下拉框只在“**双卡扩显存｜高清模式**”下显示并生效。

- `SAFE｜保守·最稳`
- `RESIDENT4｜驻留+·多用显存`
- `RESIDENT8｜高驻留·减少搬运`
- `MLP36｜副卡36%·轻度加活`
- `MLP40｜副卡40%·高负载`
- `HEAD20｜注意力+·副卡多分担`
- `BALANCE｜均衡·推荐实验`
- `MAX_TEST｜极限·可能爆显存`

下拉框只保留人话说明。实际 MLP 比例、Attention heads、trim interval、VRAM reserve 与 hot cache 会输出到控制台日志，方便提交测试结果。

## 普通 Style / 角色 LoRA

使用公开节点 `H3VM Style LoRA Stack｜普通LoRA叠加`。它与 Turbo 加速 LoRA 分离：普通 LoRA 使用 ComfyUI `LoraLoaderModelOnly` 同类的标准 ModelPatcher weight-patch 语义，并在 H3VM 建立双卡执行树时同步到实际拥有对应权重的 root / island / helper patcher。

- SINGLE_GPU：支持
- DUAL_QUIET：支持
- DUAL_CAPACITY：支持
- DUAL_SYNC_ACCEL / Stock FullThrottle：支持普通 Style LoRA，但继续禁用 Turbo 加速 LoRA
- 每个 Stack 节点提供 4 个槽位，可通过 `previous_stack` 继续串接，因此不是最多 4 个 LoRA

已知 Larry / LightX H3 Turbo 权重必须放在 Master Loader 的 Turbo 槽，不允许伪装成普通 Style LoRA。

## Turbo LoRA

当前 Master Loader 已适配两类常见 H3 Turbo LoRA：

- Larry `minimax_h3_turbo_v4_step600_ema.safetensors`
- ModelTC / LightX2V `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`

加载器会根据 LoRA 家族选择对应 sampler 语义，并统一处理 12 / 3 video-audio sigma shift。步数统一提供 4 / 6 / 8 三档；若权重有更严格的步数契约，会在运行时提示或锁定。

## 安装

1. 将 `ComfyUI-H3-MultiGPU` 文件夹放入 `ComfyUI/custom_nodes/`。
2. 安装 `requirements.txt` 中依赖（如环境尚未具备）。
3. 重启 ComfyUI。
4. 在节点搜索中找到 `H3VM Multi-GPU Loader｜H3多卡加载器`。

覆盖升级时建议先删除旧版 `ComfyUI-H3-MultiGPU` 文件夹，再解压新版，避免旧实验文件残留。

默认节点菜单只显示公开使用节点，避免历史实验节点刷屏。开发者如需重新显示全部 Lab 节点，可在启动 ComfyUI 前设置环境变量 `H3VM_SHOW_LAB_NODES=1`。

## 推荐测试顺序

第一次测试建议保持相同 Prompt / Seed / LoRA，只切模式：

1. `单卡｜标准模式`
2. `双卡协同｜日常推荐`
3. 单卡高分辨率 OOM 后，再切 `双卡扩显存｜高清模式`

Capacity 档位测试建议先跑：

`SAFE → RESIDENT8 → MLP40 → BALANCE`

`MAX_TEST` 用于摸显存墙，OOM 属于预期实验结果之一。

## 当前验证环境

主要开发与验证环境：

- Windows 11
- RTX 5060 Ti 16GB + RTX 5060 8GB
- PCIe 5.0 x8 + x8
- 无 CUDA P2P，跨卡通过 host/RAM relay
- ComfyUI 0.33.0
- PyTorch 2.13.0 + CUDA 13.0
- DynamicVRAM / comfy-aimdo
- SageAttention

其他显卡组合欢迎提交日志。异构双卡是本项目重点场景之一。

## 项目状态

- `DUAL_QUIET`：当前日常推荐
- `DUAL_CAPACITY`：可用但仍属于实验性容量模式，尤其欢迎不同显存组合与高分辨率测试
- `DUAL_SYNC_ACCEL`：内置 Stock H3 20-step FullThrottle（Dev9.4 Snapshot Islands，实验性近似执行）

项目不会把“GPU1 更忙”当作加速成功。性能模式以 wall time / 主卡关键路径为准；Capacity 模式则以“单卡 OOM、双卡能完成”为主要成功标准。

## 帮忙测试

如果你愿意贡献不同 GPU 组合的数据，直接按 `TEST_REPORT_TEMPLATE.md` 回报即可。最有价值的是：双卡型号/显存、模式、分辨率、步数、是否完成、总耗时、两张卡专用显存峰值和音质情况。

## License

见 `LICENSE` 与 `THIRD_PARTY_NOTICES.md`。
