original refer to: https://github.com/ashleve/lightning-hydra-template/tree/main/src/models

# 1. Model/ folder structure:

original structure:
```
src/models
├── components
│   ├── __init__.py
│   └── simple_dense_net.py
├── __init__.py
├── mnist_module.py
└── README.md

```

The models/ folder is harder to generalize than data/ folder,

some examples below 

- https://github.com/ashleve/lightning-hydra-template/tree/main/src/models
```
lightning-hydra-template/src/models/
├── components
│   ├── __init__.py
│   └── simple_dense_net.py
├── __init__.py
└── mnist_module.py

```

- https://github.com/gorodnitskiy/yet-another-lightning-hydra-template/tree/main/src/modules
```
yet-another-lightning-hydra-template/src/modules/
├── components
│   ├── grad_cam.py
│   ├── __init__.py
│   └── lit_module.py
├── __init__.py
├── losses
│   ├── components
│   │   ├── focal_loss.py
│   │   ├── __init__.py
│   │   ├── margin_loss.py
│   │   └── vicreg_loss.py
│   ├── __init__.py
│   └── losses.py
├── metrics
│   ├── components
│   │   ├── classification.py
│   │   ├── __init__.py
│   │   └── segmentation.py
│   ├── eval_metrics.py
│   ├── __init__.py
│   └── metrics.py
├── models
│   ├── classification.py
│   ├── __init__.py
│   ├── module.py
│   ├── reidentification.py
│   └── simple_dense_net.py
├── multiple_module.py
└── single_module.py

```

- https://github.com/nathanpainchaud/lightning-hydra-template/tree/main/src/lightning_hydra_template/models
```
lightning-hydra-template/src/lightning_hydra_template/models
├── components
│   ├── __init__.py
│   └── simple_dense_net.py
├── __init__.py
└── mnist_module.py

```


- https://github.com/lujiarui/Str2Str/tree/main/src/models
```
Str2Str/src/models
├── diffusion_module.py
├── __init__.py
├── loss.py
├── net
│   ├── denoising_ipa.py
│   ├── __init__.py
│   ├── ipa.py
│   └── layers.py
└── score
    ├── frame.py
    ├── __init__.py
    ├── r3.py
    └── so3.py
```

- https://github.com/Junjie-Zhu/IDPFold/tree/main/src/models
```
IDPFold/src/models
├── diffusion_module.py
├── __init__.py
├── loss.py
├── net
│   ├── denoising_ipa.py
│   ├── __init__.py
│   ├── ipa.py
│   └── layers.py
└── score
    ├── frame.py
    ├── __init__.py
    ├── r3.py
    └── so3.py
```

- https://github.com/lujiarui/esmdiff/tree/main/slm/models
```
esmdiff/slm/models
├── __init__.py
├── model.py
├── net.py
└── utils.py
```

- https://github.com/amorehead/alphafold3-pytorch-lightning-hydra/tree/main/alphafold3_pytorch/models
```
alphafold3-pytorch-lightning-hydra/alphafold3_pytorch/models/
├── alphafold3_module.py
├── components
│   ├── alphafold3.py
│   ├── attention.py
│   ├── callbacks
│   │   ├── ema.py
│   │   └── lr_monitor.py
│   ├── __init__.py
│   ├── inputs.py
│   ├── nlm.py
│   └── plm.py
└── __init__.py
```

- https://github.com/yinjunbo/cfpgen/tree/main/src/byprot/models
```
cfpgen/src/byprot/models
├── __init__.py
└── lm
    ├── cfp_gen_if.py
    ├── cfp_gen.py
    ├── cond_dplm.py
    ├── cond_func_dplm.py
    ├── cond_struc_dplm.py
    ├── dplm.py
    ├── esm_cfpgen_if.py
    ├── esm_cfpgen.py
    ├── esm_dplm.py
    ├── generator.py
    ├── model_utils.py
    └── modules
        ├── cfpgen_adapter.py
        ├── dplm_adapter.py
        └── gvp_transformer_encoder.py

```

- https://github.com/BioinfoMachineLearning/FlowDock/tree/main/flowdock/models
```
FlowDock/flowdock/models/
├── components
│   ├── callbacks
│   │   └── ema.py
│   ├── cpm.py
│   ├── embedding.py
│   ├── esdm.py
│   ├── flowdock.py
│   ├── hetero_graph.py
│   ├── __init__.py
│   ├── losses.py
│   ├── mht_encoder.py
│   ├── modules.py
│   ├── noise.py
│   └── transforms.py
├── flowdock_fm_module.py
└── __init__.py

```

- https://github.com/Junjie-Zhu/StruCloze/tree/main/src/model
```
StruCloze/src/model/
├── components
│   ├── embedder.py
│   ├── primitives.py
│   └── transformer.py
├── integral.py
├── loss.py
└── optimizer.py
```

- https://github.com/klemens-floege/oneprot/tree/main/src/models
```
oneprot/src/models
├── components
│   ├── base_encoder.py
│   ├── __init__.py
│   ├── loss.py
│   ├── msa_encoder.py
│   ├── peft_checkpoint.py
│   ├── retrieval_metric.py
│   ├── sequence_encoder.py
│   ├── struct_graph_encoder.py
│   ├── struct_token_encoder.py
│   └── text_encoder.py
├── __init__.py
└── oneprot_module.py

```

