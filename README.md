# Fddet

> [FDDet: Frequency-Decoupling for Boundary
Refinement in Temporal Action Detection]

<!-- [ALGORITHM] -->

## Abstract

Temporal action detection aims to locate and classify actions in untrimmed videos. While recent works focus on designing powerful feature processors for pre-trained representations, they often overlook the inherent noise and redundancy within these features. Large-scale pre-trained video encoders tend to introduce background clutter and irrelevant semantics, leading to context confusion and imprecise boundaries. To address this, we propose a frequency-aware decoupling network that improves action discriminability by filtering out noisy semantics captured by pre-trained models. Specifically, we introduce an adaptive temporal decoupling scheme that suppresses irrelevant information while preserving fine-grained atomic action details, yielding more task-specific representations. In addition, we enhance inter-frame modeling by capturing temporal variations to better distinguish actions from background redundancy. Furthermore, we present a long-short-term category-aware relation network that jointly models local transitions and long-range dependencies, improving localization precision. The refined atomic features and frequency-guided dynamics are fed into a standard detection head to produce accurate action predictions. Extensive experiments on THUMOS14, HACS, and ActivityNet-1.3 show that our method, powered by InternVideo2-6B features, achieves state-of-the-art performance on temporal action detection benchmarks.

## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train Fddet on THUMOS dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/fddet/thumos_internvideo2.py
```

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test Fddet on THUMOS dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/fddet/thumos_internvideo2.py --checkpoint exps/thumos/fddet/gpu1_id0/checkpoint/epoch_25.pth
```

## Acknowledgement

This project is developed based on [OpenTAD](https://github.com/sming256/OpenTAD). We sincerely thank the authors of OpenTAD, ActionFormer, and TriDet for their open-sourced contributions, which provide valuable foundations for our work.

For details on the training framework, dataset preparation, and usage instructions, please refer to the official OpenTAD repository:  
👉 https://github.com/sming256/OpenTAD/tree/main