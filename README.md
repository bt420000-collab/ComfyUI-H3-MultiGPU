# H3VM Multi-GPU Loader for ComfyUI

MiniMax H3 多卡加载与显存调度节点。目标不是提供一条固定工作流，而是把 H3 的模型加载、Turbo LoRA、步数、时长、分辨率、GPU 模式和 Video VAE 调度收进一个统一入口，方便接入任意 H3 工作流。

> 当前版本：v0.20.0-rc1

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

### 双卡同步加速｜需外部后端

预留接口。当前版本不会静默降级，未接入兼容的同步多卡后端时会明确报错。

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
- `DUAL_SYNC_ACCEL`：接口已预留，后端待接入

项目不会把“GPU1 更忙”当作加速成功。性能模式以 wall time / 主卡关键路径为准；Capacity 模式则以“单卡 OOM、双卡能完成”为主要成功标准。

## 帮忙测试

如果你愿意贡献不同 GPU 组合的数据，直接按 `TEST_REPORT_TEMPLATE.md` 回报即可。最有价值的是：双卡型号/显存、模式、分辨率、步数、是否完成、总耗时、两张卡专用显存峰值和音质情况。

## License

见 `LICENSE` 与 `THIRD_PARTY_NOTICES.md`。
