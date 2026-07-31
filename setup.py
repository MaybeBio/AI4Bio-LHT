#!/usr/bin/env python

# original refer to: https://github.com/ashleve/lightning-hydra-template/blob/main/setup.py
# modify it if you want, refer to:
# https://github.com/Junjie-Zhu/IDPFold/blob/main/setup.py
# https://github.com/lujiarui/Str2Str/blob/main/setup.py

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Describe Your Cool Project",
    author="",
    author_email="",
    url="https://github.com/user/project",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
