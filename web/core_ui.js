import { app } from "../../scripts/app.js";

// Label-only bilingual UI. Intentionally never reorders widgets, changes combo
// backend values, or overrides serialization.
const LABELS = {
    H3VMCoreEngine: {
        zh: {title:"H3VM Core｜模型执行引擎",ui_language:"界面语言",model:"模型",multi_gpu_enabled:"多卡支持",dual_vae_enabled:"双卡 VAE 加速",gpu_mode:"运行模式",mode4_predictor:"加速算法",gpu_participation:"副卡参与度",custom_participation:"自定义参与度",expected_steps_hint:"采样步数提示（非边界）",telemetry:"运行监测",mode:"模式",vae_policy:"VAE 策略"},
        en: {title:"H3VM Core | Model Engine",ui_language:"Language",model:"Model",multi_gpu_enabled:"Multi-GPU",dual_vae_enabled:"Dual-GPU VAE",gpu_mode:"Mode",mode4_predictor:"Acceleration",gpu_participation:"Secondary GPU Load",custom_participation:"Custom Load",expected_steps_hint:"Steps Hint (not boundary)",telemetry:"Runtime Monitor",mode:"Mode",vae_policy:"VAE Policy"},
    },
    H3VMModeAwareVideoVAEDecode: {
        zh: {title:"H3VM Video VAE｜双卡加速解码",ui_language:"界面语言",samples:"Latent",vae:"Video VAE",mode:"H3VM 模式",vae_policy:"VAE 加速策略",vae_name:"副卡 VAE 文件（需同源）",images:"图像"},
        en: {title:"H3VM Video VAE | Dual-GPU Decode",ui_language:"Language",samples:"Latent",vae:"Video VAE",mode:"H3VM Mode",vae_policy:"VAE Acceleration Policy",vae_name:"Secondary VAE File (same weights)",images:"Images"},
    },
};
function applyLabels(node,nodeName){const languageWidget=node.widgets?.find((w)=>w.name==="ui_language");const lang=languageWidget?.value==="English"?"en":"zh";const dict=LABELS[nodeName]?.[lang];if(!dict)return;node.title=dict.title;for(const widget of node.widgets||[]){if(dict[widget.name])widget.label=dict[widget.name]}for(const input of node.inputs||[]){if(dict[input.name])input.label=dict[input.name]}for(const output of node.outputs||[]){if(dict[output.name])output.label=dict[output.name]}for(const widget of node.widgets||[]){if(!["multi_gpu_enabled","telemetry","dual_vae_enabled"].includes(widget.name))continue;widget.options=widget.options||{};widget.options.label_on=lang==="zh"?"开启":"On";widget.options.label_off=lang==="zh"?"关闭":"Off";widget.options.on=widget.options.label_on;widget.options.off=widget.options.label_off}node.setDirtyCanvas?.(true,true)}
app.registerExtension({name:"H3VM.CoreSafeBilingualUI",beforeRegisterNodeDef(nodeType,nodeData){const nodeName=nodeData?.name;if(!LABELS[nodeName])return;const originalCreated=nodeType.prototype.onNodeCreated;const originalConfigure=nodeType.prototype.onConfigure;const install=function(){const languageWidget=this.widgets?.find((w)=>w.name==="ui_language");if(languageWidget&&!languageWidget.__h3vmLanguageHook){const previous=languageWidget.callback;languageWidget.callback=function(){const result=previous?.apply(this,arguments);applyLabels(languageWidget.__h3vmOwnerNode,nodeName);return result};languageWidget.__h3vmOwnerNode=this;languageWidget.__h3vmLanguageHook=true}applyLabels(this,nodeName)};nodeType.prototype.onNodeCreated=function(){const result=originalCreated?.apply(this,arguments);install.call(this);return result};nodeType.prototype.onConfigure=function(){const result=originalConfigure?.apply(this,arguments);install.call(this);return result}}});
