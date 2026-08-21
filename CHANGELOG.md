# Changelog

## main｜H3VM Core preview

- 新增 `h3vm.core_adapter`，把 GPU 模式策略从具体 Loader 中抽成共享 Core。
- 现有 Master Loader 通过运行时 bridge 继续保留原 UI 和 Larry / LightX 专用兼容路径，但模式参数改为共用 Core。
- 新增内部 `adapt_model(model, config=H3VMCoreConfig(...))` 通用 MODEL 接口。
- 通用 MODEL 首版支持 clean H3 与标准 ModelPatcher weight-patch LoRA；会把 patch 状态镜像到 H3VM block/helper patcher。
- runtime injections / object patches / hook patches / weight-wrapper patches 暂时 fail-closed，避免 GPU0/GPU1 数学路径不一致。
- 通用 MODEL 使用 ComfyUI 官方 `deepclone_multigpu()` 创建独立模型副本，不直接改写上游 MODEL。
- 公共节点数量不增加，普通用户继续使用 `H3VM Multi-GPU Loader｜H3多卡加载器`。

## v0.20.0-rc1

- 产品入口更名为 `H3VM Multi-GPU Loader｜H3多卡加载器`。
- GPU 模式改为短中文说明，技术细节移到日志。
- Capacity 实验档位改为简洁中文说明，同时保留 v0.18 / v0.19 工作流别名兼容。
- Capacity 档位仅在 `DUAL_CAPACITY` 下显示并生效。
- Custom Width / Height 仅在 `CUSTOM｜自定义` 分辨率下显示。
- Prompt 保持最下方折叠，展开后提供 420px 大输入区。
- 删除节点内额外说明板与彩色帮助文字。
- README 改为面向公开发布的安装与测试说明。
- 明确记录：当前测试机上 Capacity 8-step 推荐；4-step 存在音质风险。
