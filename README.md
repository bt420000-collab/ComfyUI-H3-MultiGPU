# H3 VRAM Master v0.21.1 for ComfyUI

**H3VM = H3 VRAM Master**  
**中文** | [English](README_EN.md)

H3VM 是为 **MiniMax H3 本地生成**设计的显存与多 GPU 执行引擎。

> **你负责工作流，H3VM 负责安排显卡。**

## 实机基准测试

📊 **[H3VM v0.21.1 单卡 / 多卡完整基准测试报告（2026-09-16）](H3VM_v0.21.1_MULTIGPU_BENCHMARK_2026-09-16.md)**

双 RTX 5060 Ti 16GB、ComfyUI 0.35.0、124 帧 / 20 Steps 实测：0.7MP 矩阵下双卡极速最高达到 **1.63× 端到端加速**，采样器步时从 **23.80s/it** 降至约 **12.8s/it**；双卡扩容模式将主卡显存峰值压低至 **6.08 GiB**。报告包含 7 种模式、两档分辨率、显存 / 利用率、逐节点耗时以及逐帧 PSNR 对比。

v0.21.1 的重点不是继续堆按钮，而是把 H3VM 收敛成更容易接入、更不容易污染其它工作流的基础执行层。

## v0.21.1 主要更新

### 1. 增强新版 ComfyUI 多卡兼容

双卡运行时请让 ComfyUI 在启动阶段看到全部 GPU：

```bash
--cuda-device all
```

这是双卡使用的关键启动要求。除此之外，H3VM 不要求你为了插件额外重配一套 ComfyUI 启动参数；原有参数通常可以继续使用，前提是两个目标 CUDA 设备确实对当前 Python / ComfyUI 进程可见。

H3VM 支持同型号和异构显卡组合，显卡名称、显存容量不要求完全一致。

### 2. 新增独立运行隔离

H3VM runtime 改为按需激活。

- 不执行 H3VM 节点时，不主动接管标准 ComfyUI 工作流；
- Prompt、Seed、分辨率、时长、Sampler、Sigmas、实际 Steps、CLIP / 文本编码器仍由原工作流负责；
- 多套 H3 / LTX / 图片工作流共存在同一个 ComfyUI 环境时，减少互相污染。

### 3. 双卡模式精简为三种实用模式

过去实验性质较强的入口已经收掉。现在只保留：

- **双卡极速**：速度优先，适合批量生成和试镜；
- **双卡扩容**：显存优先，目标是把单卡容易 OOM 的任务跑起来；
- **双卡后台**：给桌面、浏览器、剪辑软件等前台任务保留 GPU 余量，避免整机被 H3 长时间吃满。

关闭“多卡支持”即回到单卡。

### 4. 极速模式新增预测算法

双卡极速现在提供：

- **SPECTRAL｜频谱极速**：新版默认预测路线，追求更高速度上限；
- **LINEAR｜标准极速**：保留线性预测，适合做内容兼容和 A/B 对照。

不同运动强度、镜头变化和画面内容可以选择不同预测方式，不再把一套 predictor 硬套所有视频。

### 5. 新增副卡参与度 / 快速计算比例

可选：

- `100%`
- `75%`
- `50%`
- `25%`
- `自定义 1–100%`

用于适配同型号双卡、强弱异构卡、16G + 8G、16G + 16G 等不同组合。不同模式会把这个比例映射到各自的安全调度参数，而不是简单粗暴地按固定比例切所有计算。

### 6. 去掉兼容性较差的“大而全”节点

根据老用户反馈，H3VM 不再试图接管整套生成参数，正式公共入口收敛为两颗基础节点：

- **H3VM Core｜模型执行引擎**
- **H3VM Video VAE｜双卡加速解码**

主模型接法：

```text
原 Model Loader
      ↓
H3VM Core
      ↓
原 Sampler / 原工作流
```

也就是把 H3VM 当作 **Loader 后面的计算引擎**。原工作流怎么写 Prompt、怎么选 Sampler、怎么控制 Steps，都不用为了多卡重做。

## 最快上手

1. 将仓库放入 `ComfyUI/custom_nodes/`。
2. 双卡用户启动 ComfyUI 时使用 `--cuda-device all`。
3. 把原 `Model Loader -> 后续节点` 改成 `Model Loader -> H3VM Core -> 后续节点`。
4. 第一次建议：**双卡极速 + SPECTRAL + 副卡 100%**。
5. MiniMax H3 标准配置建议保持 **20 Steps / H3VM Hint 20**。
6. 显存不够切“双卡扩容”；还要同时刷 B 站、剪视频或保持桌面流畅，切“双卡后台”。

## Video VAE

需要双卡 Video VAE 时，把 H3VM Video VAE 接到原解码位置，并选择与主 VAE **同一套 MiniMax H3 Video VAE 权重**。v0.21.1 会在可验证时做轻量权重签名检查，明显选错 VAE 会直接拒绝。

双卡性能差距很大时，可以关闭 Core 中的“双卡 VAE 加速”，主模型仍可继续使用 H3VM 双卡模式。

## v0.21.1 内部稳定性修复

- Mode4 采样边界改由 ComfyUI `OUTER_SAMPLE` 生命周期管理，Steps Hint 不再承担运行结束判定；
- Predictor 首拍 timestep 初始化更干净，减少无意义坐标历史重置；
- 双卡 Video VAE 增加同源权重校验；
- 正式发布参考工作流统一为标准 **20 Steps / Hint 20**；
- 保留并合并新版 ComfyUI 下的多 GPU 可见性与异构显卡预检修复。

## 参考工作流

正式 **Release ZIP** 内附 `01_H3VM_Core_Reference_Workflow.json`。源码仓库保持精简，不再保留旧版 Quiet / Capacity 示例工作流。

## 当前兼容基线

v0.21.1 按 **ComfyUI v0.35.0** 当前接口完成发布前兼容检查。

Windows 无 NVLink / 无 CUDA P2P 的普通消费级双卡仍是 H3VM 的主要目标环境之一。

## 作者 / Author

- GitHub: `bt420000-collab`
- Bilibili: https://space.bilibili.com/16826253

## License

见 `LICENSE` 与 `THIRD_PARTY_NOTICES.md`。
