original refer to: https://github.com/ashleve/lightning-hydra-template/tree/main/src/data

# 1. Dataset vs DataModule 

original structure: 
```
src/data
├── components
│   ├── dataset.py
│   └── __init__.py
├── __init__.py
├── mnist_datamodule.py
└── README.md

```

* Inside components(or whatever): `Dataset`, A Dataset stores individual data samples and labels
* Outside components: DataModule, `DataModule` in PyTorch Lightning is a higher-level class that packages Datasets and DataLoaders together into a reusable structure

# 2. MODIFY FILES BELOW TO MATCH YOUR PROJECT!

> How to customize your own src/data/?

refer to 
* https://github.com/lujiarui/Str2Str/blob/main/src/data/components/dataset.py
* https://github.com/Junjie-Zhu/IDPFold2/blob/main/src/data/dataset.py

For example, change to your dataset, you can :
- create new `Dataset class` inside components/
- Modify your `DataModule class` outside components/
- Modify your `configs/data/mnist.yaml` (or whatever)



## We provide several sample scripts for `Dataset and DataModule` for your reference

- protein_datamodule.py: refer to https://github.com/lujiarui/Str2Str/blob/main/src/data/protein_datamodule.py