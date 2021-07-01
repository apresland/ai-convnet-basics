import os
import sys

from data.generator import DataGenerator
from models.mobilenet2 import MobileNet2Wrapper
from models.resnet50 import ResNet50Wrapper
from models.densenet169 import DenseNet169Wrapper
from models.cancernet import CancerNet
from trainers.trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 

def main():

    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    create_dirs([
        config.callbacks.tensorboard_log_dir,
        config.callbacks.evaluation_log_dir,
        config.callbacks.checkpoint_dir])

    print('Create the data generator.')
    data = DataGenerator(config)

    models = {
        "cancernet": CancerNet(),
        "densenet169": DenseNet169Wrapper(),
        "mobilenet2": MobileNet2Wrapper(),
        "resnet50": ResNet50Wrapper(),
    }

    model = models.get(config.model.name, "ERROR: invalid model name")

    print("Train classifier:",  model.__class__)
    trainer = Trainer(model, data.train_data_generator, data.valid_data_generator, config)
    trainer.train()

    model.load_weights(
        filepath=os.path.join(config.callbacks.checkpoint_dir, '%s.h5' % config.exp.name))

    loss, accuracy, precision, recall, auc = model.evaluate(
        data.test_data_generator)
    
    predictions = model.predict(
        data.test_data_generator,
        steps=data.test_data_generator.n,
        verbose=1)

    fpr, tpr, thresholds = roc_curve(
        data.test_data_generator.classes,
        predictions)

    np.savez(
        file=os.path.join(config.callbacks.evaluation_log_dir, '%s-metrics.npz' % config.exp.name), 
        loss=loss, accuracy=accuracy, precision=precision, recall=recall, auc=auc,
        fpr=fpr, tpr=tpr, thresholds=thresholds)


if __name__ == '__main__':
    main()