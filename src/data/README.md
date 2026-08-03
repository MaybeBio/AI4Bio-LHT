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

# 3. For DataModule

```
A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html

```

# 4. Collect `Dataset and DataModule` scripts to enrich the src/data folder (🌟)

- src/data/protein_datamodule.py: refer to https://github.com/lujiarui/Str2Str/blob/main/src/data/protein_datamodule.py  
- 