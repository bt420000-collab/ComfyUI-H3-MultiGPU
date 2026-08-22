# H3VM rc5 Public Core Engine

## Product boundary

- `H3VM Multi-GPU Loader｜H3多卡加载器`: standalone cockpit. Existing integrated controls remain.
- `H3VM Core｜多卡执行引擎`: external-workflow engine. `MODEL -> H3VM -> MODEL`.

## Core does not own

Prompt, Seed, width/height, duration, Turbo selection, ordinary LoRA selection, sampler, sigmas/scheduler, or actual sampling step count.

`expected_steps_hint` is only an execution hint for prefetch/final-step scheduling. It never rewires or changes the workflow sampler.

## Current generic MODEL compatibility

Supported: clean H3 ModelPatcher and standard weight-patch ModelPatcher state such as common `LoraLoaderModelOnly`.

Fail-closed for now: runtime injections, object patches, hook patches, weight-wrapper patches. These require explicit helper-GPU remapping before H3VM can guarantee both GPUs execute the same model.

## Recommended insertion point

`Base/route -> accelerator/sigma -> ordinary LoRA(s) -> H3VM Core -> Guider/Scheduler/Sampler`

For the current compatibility smoke test, begin with Clean H3 / ordinary LoRA paths. Larry runtime injections and SigmaShift object-patch compatibility are later adapters, not reasons to put business policy back into Core.
