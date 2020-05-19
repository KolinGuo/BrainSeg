import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics
from typing import List

class SparseMeanIoU(metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

class SparseConfusionMatrix(metrics.MeanIoU):
    """Computes confusion matrix"""
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.total_cm

class SparseIoU(metrics.MeanIoU):
    def __init__(self, class_idx, num_classes, name=None, dtype=None):
        super(SparseIoU, self).__init__(num_classes=num_classes, 
                name=name, dtype=dtype)

        self.class_idx = class_idx    # Computes this IoU only
        assert self.class_idx >= 0 and self.class_idx < num_classes, \
                f"Invalid class_idx {self.class_idx} " \
                f"for num_classes {self.num_classes}"

    @staticmethod
    def get_iou_metrics(num_classes, class_names) -> List["SparseIoU"]:
        return [SparseIoU(class_idx=i, num_classes=num_classes, 
            name=f'IoU/Class{i}_{class_names[i]}') for i in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

    def result(self):
        sum_over_row = tf.cast(
                tf.math.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = tf.cast(
                tf.math.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = tf.cast(
                tf.linalg.diag_part(self.total_cm), dtype=self._dtype)
                                        
        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives

        iou = tf.math.divide_no_nan(true_positives, denominator)

        return iou[self.class_idx]
