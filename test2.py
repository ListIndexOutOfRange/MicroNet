import os
import torch
from model import Model
from trainer import Trainer
import config as cfg


def build():
    print('Building Model...')
    model = Model(cfg.model['net'])
    trainer = Trainer(model)
    print(trainer)
    return model, trainer


model, trainer = build()
trainer.run()