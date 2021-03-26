
import os
import numpy as np

from base.base_trainer import BaseTrainer
from tensorflow.keras import optimizers, losses, metrics
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger
from utils.args import get_args

class Trainer(BaseTrainer):
    def __init__(self, model, train_data_generator, valid_data_generator, config):
        super(Trainer, self).__init__(model, train_data_generator, valid_data_generator, config)
        self.train_steps = self.train_data_generator.n // self.train_data_generator.batch_size
        self.valid_steps = self.valid_data_generator.n // self.valid_data_generator.batch_size
        self.callbacks = []
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s.h5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                verbose=self.config.callbacks.verbose,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                save_best_only=self.config.callbacks.checkpoint_save_best_only)
        )
        #self.callbacks.append(
        #    EarlyStopping(
        #        monitor=self.config.callbacks.early_stopping_monitor,
        #        patience=self.config.callbacks.early_stopping_patience,
        #        verbose=self.config.callbacks.verbose,
        #        restore_best_weights=True)
        #)
        self.callbacks.append(
            ReduceLROnPlateau(
                monitor=self.config.callbacks.reduce_on_plateau_monitor,
                factor=self.config.callbacks.reduce_on_plateau_factor,
                patience=self.config.callbacks.reduce_on_plateau_patience, 
                verbose=self.config.callbacks.verbose)
        )
        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph)
        )
        self.callbacks.append(
            CSVLogger(
                filename=os.path.join(self.config.callbacks.evaluation_log_dir, '%s.csv' % self.config.exp.name))
        )


    def train(self):
        self.model.compile(
            optimizer=optimizers.Adam(lr=1e-3),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                metrics.Precision(),
                metrics.Recall(),
                metrics.AUC()])
        self.model.fit(
            self.train_data_generator,
            epochs=self.config.trainer.epochs,
            steps_per_epoch=self.train_steps,
            validation_data=self.valid_data_generator,
            validation_steps=self.valid_steps,
            callbacks=self.callbacks,
            verbose=1)

   