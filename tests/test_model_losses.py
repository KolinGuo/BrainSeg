"""Test Cases for Custom losses based on tf.keras.losses"""

import sys
import unittest
from typing import List

import numpy as np
from tensorflow.keras import losses
from src.models.losses import BalancedSCCE, SparseFocalLoss, \
        BalancedSparseFocalLoss
sys.path.append("..")

class TestBalancedSCCE(unittest.TestCase):
    """Testing the Balanced SCCE implementation"""
    def setUp(self):
        """Setup shared by all tests"""
        self.scce = losses.SparseCategoricalCrossentropy()
        self.bscce_equal = BalancedSCCE([1, 1, 1])

        self.class_weights = [0.2, 0.3, 0.5]
        self.bscce_unequal = BalancedSCCE(self.class_weights)

    def test_equal_weights_1d(self):
        """Testing equal weights for 1D data"""
        y_true = [1, 2]
        y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
        scce = self.scce(y_true, y_pred).numpy()
        bscce = self.bscce_equal(y_true, y_pred).numpy()
        self.assertEqual(scce, bscce)

    def test_unequal_weights_1d(self):
        """Testing unequal weights for 1D data"""
        y_true = [1, 2]
        y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
        sample_weight = np.take(self.class_weights, y_true)
        scce = self.scce(y_true, y_pred, sample_weight=sample_weight).numpy()
        bscce = self.bscce_unequal(y_true, y_pred).numpy()
        self.assertEqual(scce, bscce)

    def test_equal_weights_reduction_1d(self):
        """Testing equal weights reductions for 1D data"""
        y_true = [1, 2]
        y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

        self.scce = losses.SparseCategoricalCrossentropy(
            reduction=losses.Reduction.SUM)
        self.bscce_equal = BalancedSCCE([1, 1, 1],
                                        reduction=losses.Reduction.SUM)
        scce = self.scce(y_true, y_pred).numpy()
        bscce = self.bscce_equal(y_true, y_pred).numpy()
        np.testing.assert_array_equal(scce, bscce)

        self.scce = losses.SparseCategoricalCrossentropy(
            reduction=losses.Reduction.NONE)
        self.bscce_equal = BalancedSCCE([1, 1, 1],
                                        reduction=losses.Reduction.NONE)
        scce = self.scce(y_true, y_pred).numpy()
        bscce = self.bscce_equal(y_true, y_pred).numpy()
        np.testing.assert_array_equal(scce, bscce)

    def test_unequal_weights_reduction_1d(self):
        """Testing unequal weights reductions for 1D data"""
        y_true = [1, 2]
        y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
        sample_weight = np.take(self.class_weights, y_true)

        self.scce = losses.SparseCategoricalCrossentropy(
            reduction=losses.Reduction.SUM)
        self.bscce_unequal = BalancedSCCE(self.class_weights,
                                          reduction=losses.Reduction.SUM)
        scce = self.scce(y_true, y_pred, sample_weight=sample_weight).numpy()
        bscce = self.bscce_unequal(y_true, y_pred).numpy()
        np.testing.assert_array_equal(scce, bscce)

        self.scce = losses.SparseCategoricalCrossentropy(
            reduction=losses.Reduction.NONE)
        self.bscce_unequal = BalancedSCCE(self.class_weights,
                                          reduction=losses.Reduction.NONE)
        scce = self.scce(y_true, y_pred, sample_weight=sample_weight).numpy()
        bscce = self.bscce_unequal(y_true, y_pred).numpy()
        np.testing.assert_array_equal(scce, bscce)

    def test_equal_weights_logits_1d(self):
        """Testing equal weights logits for 1D data"""
        y_true = [1, 2]
        y_pred = [[-0.05, 0.3, 0.19], [0.2, -0.4, 0.12]]

        self.scce = losses.SparseCategoricalCrossentropy(from_logits=True)
        self.bscce_equal = BalancedSCCE([1, 1, 1], from_logits=True)
        scce = self.scce(y_true, y_pred).numpy()
        bscce = self.bscce_equal(y_true, y_pred).numpy()
        np.testing.assert_array_equal(scce, bscce)

    def test_unequal_weights_logits_1d(self):
        """Testing unequal weights logits for 1D data"""
        y_true = [1, 2]
        y_pred = [[-0.05, 0.3, 0.19], [0.2, -0.4, 0.12]]
        sample_weight = np.take(self.class_weights, y_true)

        self.scce = losses.SparseCategoricalCrossentropy(from_logits=True)
        self.bscce_unequal = BalancedSCCE(self.class_weights, from_logits=True)
        scce = self.scce(y_true, y_pred, sample_weight=sample_weight).numpy()
        bscce = self.bscce_unequal(y_true, y_pred).numpy()
        np.testing.assert_array_equal(scce, bscce)

    def test_equal_weights_2d(self):
        """Testing equal weights for 2D data"""
        y_true = [[1, 2], [0, 2]]
        y_pred = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
                  [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]]]
        scce = self.scce(y_true, y_pred).numpy()
        bscce = self.bscce_equal(y_true, y_pred).numpy()
        self.assertEqual(scce, bscce)

    def test_unequal_weights_2d(self):
        """Testing unequal weights for 2D data"""
        y_true = [[1, 2], [0, 2]]
        y_pred = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
                  [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]]]
        sample_weight = np.take(self.class_weights, y_true)
        scce = self.scce(y_true, y_pred, sample_weight=sample_weight).numpy()
        bscce = self.bscce_unequal(y_true, y_pred).numpy()
        self.assertEqual(scce, bscce)

    def test_equal_weights_reduction_2d(self):
        """Testing equal weights reductions for 2D data"""
        y_true = [[1, 2], [0, 2]]
        y_pred = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
                  [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]]]

        self.scce = losses.SparseCategoricalCrossentropy(
            reduction=losses.Reduction.SUM)
        self.bscce_equal = BalancedSCCE([1, 1, 1],
                                        reduction=losses.Reduction.SUM)
        scce = self.scce(y_true, y_pred).numpy()
        bscce = self.bscce_equal(y_true, y_pred).numpy()
        np.testing.assert_array_equal(scce, bscce)

        self.scce = losses.SparseCategoricalCrossentropy(
            reduction=losses.Reduction.NONE)
        self.bscce_equal = BalancedSCCE([1, 1, 1],
                                        reduction=losses.Reduction.NONE)
        scce = self.scce(y_true, y_pred).numpy()
        bscce = self.bscce_equal(y_true, y_pred).numpy()
        np.testing.assert_array_equal(scce, bscce)

    def test_unequal_weights_reduction_2d(self):
        """Testing unequal weights reductions for 2D data"""
        y_true = [[1, 2], [0, 2]]
        y_pred = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
                  [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]]]
        sample_weight = np.take(self.class_weights, y_true)

        self.scce = losses.SparseCategoricalCrossentropy(
            reduction=losses.Reduction.SUM)
        self.bscce_unequal = BalancedSCCE(self.class_weights,
                                          reduction=losses.Reduction.SUM)
        scce = self.scce(y_true, y_pred, sample_weight=sample_weight).numpy()
        bscce = self.bscce_unequal(y_true, y_pred).numpy()
        np.testing.assert_array_equal(scce, bscce)

        self.scce = losses.SparseCategoricalCrossentropy(
            reduction=losses.Reduction.NONE)
        self.bscce_unequal = BalancedSCCE(self.class_weights,
                                          reduction=losses.Reduction.NONE)
        scce = self.scce(y_true, y_pred, sample_weight=sample_weight).numpy()
        bscce = self.bscce_unequal(y_true, y_pred).numpy()
        np.testing.assert_array_equal(scce, bscce)

    def test_equal_weights_logits_2d(self):
        """Testing equal weights logits for 2D data"""
        y_true = [[1, 2], [0, 2]]
        y_pred = [[[-0.05, 0.3, 0.19], [0.2, -0.4, 0.12]],
                  [[-0.1, 0.22, -0.73], [0.23, -0.52, 0.2]]]

        self.scce = losses.SparseCategoricalCrossentropy(from_logits=True)
        self.bscce_equal = BalancedSCCE([1, 1, 1], from_logits=True)
        scce = self.scce(y_true, y_pred).numpy()
        bscce = self.bscce_equal(y_true, y_pred).numpy()
        np.testing.assert_array_equal(scce, bscce)

    def test_unequal_weights_logits_2d(self):
        """Testing unequal weights logits for 2D data"""
        y_true = [[1, 2], [0, 2]]
        y_pred = [[[-0.05, 0.3, 0.19], [0.2, -0.4, 0.12]],
                  [[-0.1, 0.22, -0.73], [0.23, -0.52, 0.2]]]
        sample_weight = np.take(self.class_weights, y_true)

        self.scce = losses.SparseCategoricalCrossentropy(from_logits=True)
        self.bscce_unequal = BalancedSCCE(self.class_weights, from_logits=True)
        scce = self.scce(y_true, y_pred, sample_weight=sample_weight).numpy()
        bscce = self.bscce_unequal(y_true, y_pred).numpy()
        np.testing.assert_array_equal(scce, bscce)

def get_one_hot(targets: List[int], num_classes=3) -> "NDArray[bool]":
    """Convert a sparse target into one-hot format"""
    targets = np.array(targets)
    res = np.eye(num_classes)[targets.reshape(-1)]
    return res.reshape(list(targets.shape)+[num_classes]).astype('bool')

def softmax(logit: List[float], axis=-1):
    """Softmax function"""
    prop = np.exp(logit - np.max(logit, axis, keepdims=True))
    return prop / np.sum(prop, axis, keepdims=True)

class TestSparseFocalLoss(unittest.TestCase):
    """Testing the sparse focal loss implementation"""
    def setUp(self):
        """Setup shared by all tests"""
        self.scce = losses.SparseCategoricalCrossentropy(
            reduction=losses.Reduction.NONE)
        self.gamma1 = 2.0
        self.gamma2 = 4.0
        self.sfl1 = SparseFocalLoss(gamma=self.gamma1)
        self.sfl2 = SparseFocalLoss(gamma=self.gamma2)

    def test_1d(self):
        """Testing for 1D data"""
        y_true = [1, 2]
        y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
        scce = self.scce(y_true, y_pred).numpy()

        scce1 = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma1
        scce1 = scce1.mean()
        sfl1 = self.sfl1(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce1, sfl1, rtol=1e-7)
        scce2 = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma2
        scce2 = scce2.mean()
        sfl2 = self.sfl2(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce2, sfl2, rtol=1e-6)

    def test_reduction_1d(self):
        """Testing reductions for 1D data"""
        y_true = [1, 2]
        y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
        scce = self.scce(y_true, y_pred).numpy()

        self.sfl1 = SparseFocalLoss(gamma=self.gamma1,
                                    reduction=losses.Reduction.SUM)
        self.sfl2 = SparseFocalLoss(gamma=self.gamma2,
                                    reduction=losses.Reduction.SUM)

        scce1 = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma1
        scce1 = scce1.sum()
        sfl1 = self.sfl1(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce1, sfl1, rtol=1e-7)
        scce2 = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma2
        scce2 = scce2.sum()
        sfl2 = self.sfl2(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce2, sfl2, rtol=1e-6)

        self.sfl1 = SparseFocalLoss(gamma=self.gamma1,
                                    reduction=losses.Reduction.NONE)
        self.sfl2 = SparseFocalLoss(gamma=self.gamma2,
                                    reduction=losses.Reduction.NONE)

        scce1 = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma1
        sfl1 = self.sfl1(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce1, sfl1, rtol=1e-5)
        scce2 = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma2
        sfl2 = self.sfl2(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce2, sfl2, rtol=1e-5)

    def test_logits_1d(self):
        """Testing logits for 1D data"""
        y_true = [1, 2]
        y_pred = [[-0.05, 0.3, 0.19], [0.2, -0.4, 0.12]]

        self.scce = losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=losses.Reduction.NONE)
        scce = self.scce(y_true, y_pred).numpy()

        self.sfl1 = SparseFocalLoss(gamma=self.gamma1,
                                    from_logits=True)
        self.sfl2 = SparseFocalLoss(gamma=self.gamma2,
                                    from_logits=True)

        scce1 = scce * \
            (1 - np.where(get_one_hot(y_true), softmax(y_pred), 0).sum(axis=-1))\
            **self.gamma1
        scce1 = scce1.mean()
        sfl1 = self.sfl1(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce1, sfl1, rtol=1e-7)
        scce2 = scce * \
            (1 - np.where(get_one_hot(y_true), softmax(y_pred), 0).sum(axis=-1))\
            **self.gamma2
        scce2 = scce2.mean()
        sfl2 = self.sfl2(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce2, sfl2, rtol=1e-6)

    def test_2d(self):
        """Testing for 2D data"""
        y_true = [[1, 2], [0, 2]]
        y_pred = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
                  [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]]]
        scce = self.scce(y_true, y_pred).numpy()

        scce1 = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma1
        scce1 = scce1.mean()
        sfl1 = self.sfl1(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce1, sfl1, rtol=1e-7)
        scce2 = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma2
        scce2 = scce2.mean()
        sfl2 = self.sfl2(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce2, sfl2, rtol=1e-6)

    def test_reduction_2d(self):
        """Testing reductions for 2D data"""
        y_true = [[1, 2], [0, 2]]
        y_pred = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
                  [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]]]
        scce = self.scce(y_true, y_pred).numpy()

        self.sfl1 = SparseFocalLoss(gamma=self.gamma1,
                                    reduction=losses.Reduction.SUM)
        self.sfl2 = SparseFocalLoss(gamma=self.gamma2,
                                    reduction=losses.Reduction.SUM)

        scce1 = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma1
        scce1 = scce1.sum()
        sfl1 = self.sfl1(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce1, sfl1, rtol=1e-7)
        scce2 = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma2
        scce2 = scce2.sum()
        sfl2 = self.sfl2(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce2, sfl2, rtol=1e-6)

        self.sfl1 = SparseFocalLoss(gamma=self.gamma1,
                                    reduction=losses.Reduction.NONE)
        self.sfl2 = SparseFocalLoss(gamma=self.gamma2,
                                    reduction=losses.Reduction.NONE)

        scce1 = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma1
        sfl1 = self.sfl1(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce1, sfl1, rtol=1e-5)
        scce2 = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma2
        sfl2 = self.sfl2(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce2, sfl2, rtol=1e-5)

    def test_logits_2d(self):
        """Testing logits for 2D data"""
        y_true = [[1, 2], [0, 2]]
        y_pred = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
                  [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]]]

        self.scce = losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=losses.Reduction.NONE)
        scce = self.scce(y_true, y_pred).numpy()

        self.sfl1 = SparseFocalLoss(gamma=self.gamma1,
                                    from_logits=True)
        self.sfl2 = SparseFocalLoss(gamma=self.gamma2,
                                    from_logits=True)

        scce1 = scce * \
            (1 - np.where(get_one_hot(y_true), softmax(y_pred), 0).sum(axis=-1))\
            **self.gamma1
        scce1 = scce1.mean()
        sfl1 = self.sfl1(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce1, sfl1, rtol=1e-7)
        scce2 = scce * \
            (1 - np.where(get_one_hot(y_true), softmax(y_pred), 0).sum(axis=-1))\
            **self.gamma2
        scce2 = scce2.mean()
        sfl2 = self.sfl2(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce2, sfl2, rtol=1e-6)

class TestBalancedSparseFocalLoss(unittest.TestCase):
    """Testing the Balanced sparse focal loss implementation"""
    def setUp(self):
        """Setup shared by all tests"""
        self.scce = losses.SparseCategoricalCrossentropy(
            reduction=losses.Reduction.NONE)
        self.gamma = 4.0

        self.sfl_equal = BalancedSparseFocalLoss([1, 1, 1], gamma=self.gamma)

        self.class_weights = [0.2, 0.3, 0.5]
        self.sfl_unequal = BalancedSparseFocalLoss(self.class_weights,
                                                   gamma=self.gamma)

    def test_equal_weights_1d(self):
        """Testing equal weights for 1D data"""
        y_true = [1, 2]
        y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
        scce = self.scce(y_true, y_pred).numpy()

        scce = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma
        scce = scce.mean()
        sfl = self.sfl_equal(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce, sfl, rtol=1e-6)

    def test_unequal_weights_1d(self):
        """Testing unequal weights for 1D data"""
        y_true = [1, 2]
        y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
        sample_weight = np.take(self.class_weights, y_true)
        scce = self.scce(y_true, y_pred, sample_weight=sample_weight).numpy()

        scce = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma
        scce = scce.mean()
        sfl = self.sfl_unequal(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce, sfl, rtol=1e-6)

    def test_equal_weights_reduction_1d(self):
        """Testing equal weights reductions for 1D data"""
        y_true = [1, 2]
        y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
        scce = self.scce(y_true, y_pred).numpy()

        self.sfl_equal = BalancedSparseFocalLoss(
            [1, 1, 1],
            gamma=self.gamma,
            reduction=losses.Reduction.SUM)

        scce1 = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma
        scce1 = scce1.sum()
        sfl1 = self.sfl_equal(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce1, sfl1, rtol=1e-6)

        self.sfl_equal = BalancedSparseFocalLoss(
            [1, 1, 1],
            gamma=self.gamma,
            reduction=losses.Reduction.NONE)

        scce2 = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma
        sfl2 = self.sfl_equal(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce2, sfl2, rtol=1e-5)

    def test_unequal_weights_reduction_1d(self):
        """Testing unequal weights reductions for 1D data"""
        y_true = [1, 2]
        y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
        sample_weight = np.take(self.class_weights, y_true)
        scce = self.scce(y_true, y_pred, sample_weight=sample_weight).numpy()

        self.sfl_unequal = BalancedSparseFocalLoss(
            self.class_weights,
            gamma=self.gamma,
            reduction=losses.Reduction.SUM)

        scce1 = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma
        scce1 = scce1.sum()
        sfl1 = self.sfl_unequal(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce1, sfl1, rtol=1e-6)

        self.sfl_unequal = BalancedSparseFocalLoss(
            self.class_weights,
            gamma=self.gamma,
            reduction=losses.Reduction.NONE)

        scce2 = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma
        sfl2 = self.sfl_unequal(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce2, sfl2, rtol=1e-5)

    def test_equal_weights_logits_1d(self):
        """Testing equal weights logits for 1D data"""
        y_true = [1, 2]
        y_pred = [[-0.05, 0.3, 0.19], [0.2, -0.4, 0.12]]

        self.scce = losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=losses.Reduction.NONE)
        scce = self.scce(y_true, y_pred).numpy()

        self.sfl_equal = BalancedSparseFocalLoss(
            [1, 1, 1], gamma=self.gamma, from_logits=True)

        scce = scce * \
            (1 - np.where(get_one_hot(y_true), softmax(y_pred), 0).sum(axis=-1))\
            **self.gamma
        scce = scce.mean()
        sfl = self.sfl_equal(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce, sfl, rtol=1e-6)

    def test_unequal_weights_logits_1d(self):
        """Testing unequal weights logits for 1D data"""
        y_true = [1, 2]
        y_pred = [[-0.05, 0.3, 0.19], [0.2, -0.4, 0.12]]
        sample_weight = np.take(self.class_weights, y_true)

        self.scce = losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=losses.Reduction.NONE)
        scce = self.scce(y_true, y_pred, sample_weight=sample_weight).numpy()

        self.sfl_unequal = BalancedSparseFocalLoss(
            self.class_weights,
            gamma=self.gamma,
            from_logits=True)

        scce = scce * \
            (1 - np.where(get_one_hot(y_true), softmax(y_pred), 0).sum(axis=-1))\
            **self.gamma
        scce = scce.mean()
        sfl = self.sfl_unequal(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce, sfl, rtol=1e-6)

    def test_equal_weights_2d(self):
        """Testing equal weights for 2D data"""
        y_true = [[1, 2], [0, 2]]
        y_pred = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
                  [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]]]
        scce = self.scce(y_true, y_pred).numpy()

        scce = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma
        scce = scce.mean()
        sfl = self.sfl_equal(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce, sfl, rtol=1e-6)

    def test_unequal_weights_2d(self):
        """Testing unequal weights for 2D data"""
        y_true = [[1, 2], [0, 2]]
        y_pred = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
                  [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]]]
        sample_weight = np.take(self.class_weights, y_true)
        scce = self.scce(y_true, y_pred, sample_weight=sample_weight).numpy()

        scce = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma
        scce = scce.mean()
        sfl = self.sfl_unequal(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce, sfl, rtol=1e-6)

    def test_equal_weights_reduction_2d(self):
        """Testing equal weights reductions for 2D data"""
        y_true = [[1, 2], [0, 2]]
        y_pred = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
                  [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]]]
        scce = self.scce(y_true, y_pred).numpy()

        self.sfl_equal = BalancedSparseFocalLoss(
            [1, 1, 1],
            gamma=self.gamma,
            reduction=losses.Reduction.SUM)

        scce1 = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma
        scce1 = scce1.sum()
        sfl1 = self.sfl_equal(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce1, sfl1, rtol=1e-6)

        self.sfl_equal = BalancedSparseFocalLoss(
            [1, 1, 1],
            gamma=self.gamma,
            reduction=losses.Reduction.NONE)

        scce2 = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma
        sfl2 = self.sfl_equal(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce2, sfl2, rtol=1e-5)

    def test_unequal_weights_reduction_2d(self):
        """Testing unequal weights reductions for 2D data"""
        y_true = [[1, 2], [0, 2]]
        y_pred = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
                  [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]]]
        sample_weight = np.take(self.class_weights, y_true)
        scce = self.scce(y_true, y_pred, sample_weight=sample_weight).numpy()

        self.sfl_unequal = BalancedSparseFocalLoss(
            self.class_weights,
            gamma=self.gamma,
            reduction=losses.Reduction.SUM)

        scce1 = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma
        scce1 = scce1.sum()
        sfl1 = self.sfl_unequal(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce1, sfl1, rtol=1e-6)

        self.sfl_unequal = BalancedSparseFocalLoss(
            self.class_weights,
            gamma=self.gamma,
            reduction=losses.Reduction.NONE)

        scce2 = scce * \
            (1 - np.where(get_one_hot(y_true), y_pred, 0).sum(axis=-1))\
            **self.gamma
        sfl2 = self.sfl_unequal(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce2, sfl2, rtol=1e-5)

    def test_equal_weights_logits_2d(self):
        """Testing equal weights logits for 2D data"""
        y_true = [[1, 2], [0, 2]]
        y_pred = [[[-0.05, 0.3, 0.19], [0.2, -0.4, 0.12]],
                  [[-0.1, 0.22, -0.73], [0.23, -0.52, 0.2]]]

        self.scce = losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=losses.Reduction.NONE)
        scce = self.scce(y_true, y_pred).numpy()

        self.sfl_equal = BalancedSparseFocalLoss(
            [1, 1, 1], gamma=self.gamma, from_logits=True)

        scce = scce * \
            (1 - np.where(get_one_hot(y_true), softmax(y_pred), 0).sum(axis=-1))\
            **self.gamma
        scce = scce.mean()
        sfl = self.sfl_equal(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce, sfl, rtol=1e-6)

    def test_unequal_weights_logits_2d(self):
        """Testing unequal weights logits for 2D data"""
        y_true = [[1, 2], [0, 2]]
        y_pred = [[[-0.05, 0.3, 0.19], [0.2, -0.4, 0.12]],
                  [[-0.1, 0.22, -0.73], [0.23, -0.52, 0.2]]]
        sample_weight = np.take(self.class_weights, y_true)

        self.scce = losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=losses.Reduction.NONE)
        scce = self.scce(y_true, y_pred, sample_weight=sample_weight).numpy()

        self.sfl_unequal = BalancedSparseFocalLoss(
            self.class_weights,
            gamma=self.gamma,
            from_logits=True)

        scce = scce * \
            (1 - np.where(get_one_hot(y_true), softmax(y_pred), 0).sum(axis=-1))\
            **self.gamma
        scce = scce.mean()
        sfl = self.sfl_unequal(y_true, y_pred).numpy()
        np.testing.assert_allclose(scce, sfl, rtol=1e-6)

if __name__ == '__main__':
    unittest.main()
