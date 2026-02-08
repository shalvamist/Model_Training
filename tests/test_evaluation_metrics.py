import unittest
import numpy as np
import sys
import os

# Add root to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from library.utils import MetricsCalculator

class TestMetrics(unittest.TestCase):
    def test_regression_metrics(self):
        true = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 1.9, 3.2])
        m = MetricsCalculator.get_regression_metrics(true, pred)
        # MSE: ((0.1)^2 + (-0.1)^2 + (0.2)^2)/3 = (0.01 + 0.01 + 0.04)/3 = 0.02
        self.assertAlmostEqual(m['MSE'], 0.02, places=4)
        self.assertAlmostEqual(m['MAE'], (0.1+0.1+0.2)/3, places=4)
        
    def test_directional_accuracy(self):
        true = np.array([1.0, -1.0, 1.0, -1.0])
        pred = np.array([0.5, -0.5, -0.5, 0.5]) # TT, TT, TF, TF
        acc = MetricsCalculator.get_directional_accuracy(true, pred)
        self.assertEqual(acc, 0.5)
        
    def test_classification_metrics(self):
        true = np.array([0, 1, 2, 0])
        pred = np.array([0, 1, 2, 0])
        m = MetricsCalculator.get_classification_metrics(true, pred)
        self.assertEqual(m['Accuracy'], 1.0)
        self.assertEqual(m['MCC'], 1.0)
        
        true = np.array([0, 1, 2, 0])
        pred = np.array([0, 1, 0, 2]) # 2 wrong
        m = MetricsCalculator.get_classification_metrics(true, pred)
        self.assertEqual(m['Accuracy'], 0.5)

    def test_trading_simulation(self):
        # returns: 10%, -10%, 10%
        returns = np.array([0.1, -0.1, 0.1])
        # pred: Bull, Bear, Bull (Ideal)
        pred_class = np.array([2, 0, 2])
        
        m = MetricsCalculator.get_simple_trading_simulation(returns, pred_class)
        
        # Strategy returns: 0.1, -(-0.1)=0.1, 0.1
        # Cum: 1.1 * 1.1 * 1.1 - 1 = 1.331 - 1 = 0.331
        self.assertAlmostEqual(m['Total_Strategy_Return'], 0.331, places=4)
        self.assertEqual(m['Win_Rate'], 1.0) # 3 wins out of 3 trades
        
        # Market returns: 1.1 * 0.9 * 1.1 - 1 = 1.089 - 1 = 0.089
        self.assertAlmostEqual(m['Total_Market_Return'], 0.089, places=4)

if __name__ == '__main__':
    unittest.main()
