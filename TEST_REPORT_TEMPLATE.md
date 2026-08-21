# H3VM 测试回报模板

复制下面内容提交即可，能填多少填多少：

```text
GPU0 / VRAM:
GPU1 / VRAM:
系统:
ComfyUI 版本:
PyTorch / CUDA:

GPU 模式:
Capacity 档位（如适用）:
Turbo LoRA:
步数:
分辨率:
时长:

结果: 完成 / OOM / 报错 / 卡死
总耗时:
GPU0 专用显存峰值:
GPU1 专用显存峰值:
GPU0 利用率大致范围:
GPU1 利用率大致范围:
音质: 正常 / 变差 / 无音频
视频质量: 正常 / 异常

备注:
```

如果是报错或 OOM，请附完整 ComfyUI 控制台日志。Capacity 档位会把实际 MLP 比例、Attention head split、trim interval、reserve 和 hot cache 输出到日志，不需要手工抄内部参数。
