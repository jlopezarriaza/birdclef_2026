import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

class CompetitionAUC(tf.keras.callbacks.Callback):
    """
    Keras callback to calculate the competition-specific Macro ROC-AUC.
    Skips classes with no true positive labels in the validation set.
    """
    def __init__(self, validation_generator):
        super().__init__()
        self.val_gen = validation_generator

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # 1. Gather all predictions and true labels
        y_true = []
        y_pred = []
        for i in range(len(self.val_gen)):
            batch_x, batch_y = self.val_gen[i]
            y_true.extend(batch_y)
            y_pred.extend(self.model.predict(batch_x, verbose=0))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 2. Identify columns with at least one positive label
        solution_sums = y_true.sum(axis=0)
        scored_columns = np.where(solution_sums > 0)[0]
        
        if len(scored_columns) == 0:
            print("Warning: No positive labels found in validation set!")
            logs["val_comp_auc"] = 0.5
            return

        # 3. Calculate macro-averaged ROC-AUC only on those columns
        comp_auc = roc_auc_score(
            y_true[:, scored_columns], 
            y_pred[:, scored_columns], 
            average='macro'
        )
        
        logs["val_comp_auc"] = comp_auc
        print(f" - val_comp_auc: {comp_auc:.4f}")
