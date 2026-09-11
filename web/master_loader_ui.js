import { app } from "../../scripts/app.js";

// H3 VRAM Master Loader UI polish.
// Keep the public node compact and self-explanatory:
// - Capacity profile is visible only in DUAL_CAPACITY.
// - Custom width/height are visible only for CUSTOM resolution.
// - Prompt stays last, advanced/folded by default, with a large editor when open.
// - No extra help panel or colored/blue explanatory text is injected.
app.registerExtension({
    name: "H3VM.VRAMMasterLoaderUI",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "H3VMMasterLoader") return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        const originalConfigure = nodeType.prototype.onConfigure;

        const installUi = function () {
            const find = (name) => this.widgets?.find((w) => w.name === name);
            const modeWidget = find("gpu_mode");
            const capacityWidget = find("capacity_vram_profile");
            const resolutionWidget = find("resolution");
            const widthWidget = find("custom_width");
            const heightWidget = find("custom_height");
            const promptWidget = find("prompt");

            const refreshLayout = () => {
                this.setDirtyCanvas?.(true, true);
                requestAnimationFrame(() => {
                    try {
                        const natural = this.computeSize?.();
                        if (natural && this.size) {
                            this.setSize?.([Math.max(this.size[0], natural[0]), natural[1]]);
                        }
                    } catch (_) {
                    }
                    this.setDirtyCanvas?.(true, true);
                });
            };

            const syncCapacityVisibility = () => {
                if (!capacityWidget || !modeWidget) return;
                const active = String(modeWidget.value ?? "").startsWith("DUAL_CAPACITY");
                capacityWidget.hidden = !active;
                capacityWidget.disabled = !active;
                refreshLayout();
            };

            const syncCustomResolutionVisibility = () => {
                if (!resolutionWidget) return;
                const custom = String(resolutionWidget.value ?? "").startsWith("CUSTOM");
                for (const widget of [widthWidget, heightWidget]) {
                    if (!widget) continue;
                    widget.hidden = !custom;
                    widget.disabled = !custom;
                }
                refreshLayout();
            };

            const chainCallback = (widget, extra) => {
                if (!widget || widget.__h3vmUiChained) return;
                const previous = widget.callback;
                widget.callback = function () {
                    const result = previous?.apply(this, arguments);
                    extra();
                    return result;
                };
                widget.__h3vmUiChained = true;
            };

            chainCallback(modeWidget, syncCapacityVisibility);
            chainCallback(resolutionWidget, syncCustomResolutionVisibility);

            if (promptWidget) {
                promptWidget.options = promptWidget.options || {};
                promptWidget.options.minNodeSize = [590, 470];
                const el = promptWidget.element || promptWidget.inputEl;
                if (el) {
                    el.style.minHeight = "420px";
                    el.style.height = "420px";
                    el.style.resize = "vertical";
                    el.setAttribute("spellcheck", "false");
                }
            }

            syncCapacityVisibility();
            syncCustomResolutionVisibility();
        };

        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            installUi.call(this);
            requestAnimationFrame(() => installUi.call(this));
            setTimeout(() => installUi.call(this), 250);
            return result;
        };

        nodeType.prototype.onConfigure = function () {
            const result = originalConfigure?.apply(this, arguments);
            requestAnimationFrame(() => installUi.call(this));
            return result;
        };
    },
});