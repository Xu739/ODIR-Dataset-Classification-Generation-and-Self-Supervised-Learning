from models.model import GetModel

import os

os.chdir('..')

import yaml

from experiments.configs.par import Struct

par = yaml.safe_load(open('experiments/configs/Classification.yaml','r'))
par = Struct(**par)

model = GetModel(par)

print(model)