import unittest
import torch
import numpy as np
import pandas as pd
import tempfile
import json
import shutil
import os
import sys

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Model_trading_training.library.factory import ModelFactory
from Model_trading_training.library.trainer import ModelTrainer
from Model_trading_training.library.processors import ProcessorV11, ProcessorV13, ProcessorV15
from Model_trading_training.library.network_architectures import HybridJointNetwork, ExperimentalNetwork
from Model_trading_training.library.advanced_components import KANLinear, RevIN

class TestModelLibrary(unittest.TestCase):
    
    def setUp(self):
        # Create Dummy Data
        self.dummy_df = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=200),
            'underlying_close': np.random.rand(200) * 100 + 300,
            'underlying_open': np.random.rand(200) * 100 + 300,
            'underlying_high': np.random.rand(200) * 100 + 300,
            'underlying_low': np.random.rand(200) * 100 + 300,
            'volume': np.random.rand(200) * 1000000,
            'vix': np.random.rand(200) * 10 + 15,
            'us_treasury_10y': np.ones(200) * 4.0,
            # Extra tickers for V15
            'XLY': np.random.rand(200) * 100,
            'XLP': np.random.rand(200) * 100,
            'XLK': np.random.rand(200) * 100,
            'XLU': np.random.rand(200) * 100
        }).set_index('date')
        
        self.config = {
            "hidden_size": 32,
            "lstm_layers": 1,
            "trans_layers": 1,
            "n_experts": 2,
            "dropout": 0.0,
            "feature_cols": ["f1", "f2"], # Dummy
            "static_cols": ["s1"],
            "batch_size": 4
        }

    def test_processors_instantiation(self):
        """Test if processors can be instantiated and basic methods exist"""
        p11 = ProcessorV11({'seq_length': 10})
        p13 = ProcessorV13({'seq_length': 10})
        p15 = ProcessorV15({'seq_length': 10})
        
        self.assertIsInstance(p11, ProcessorV11)
        self.assertIsInstance(p15, ProcessorV15)
        
    def test_kan_linear(self):
        """Test Kolmogorov-Arnold Layer"""
        batch = 10
        in_dim = 16
        out_dim = 8
        kan = KANLinear(in_dim, out_dim)
        x = torch.randn(batch, in_dim)
        y = kan(x)
        self.assertEqual(y.shape, (batch, out_dim))
        
    def test_revin(self):
        """Test Reversible Instance Norm"""
        revin = RevIN(5)
        x = torch.randn(10, 20, 5) # Batch, Seq, Feat
        x_norm = revin(x, 'norm')
        x_denorm = revin(x_norm, 'denorm')
        
        # Check shapes
        self.assertEqual(x_norm.shape, x.shape)
        # Check reversibility (approx)
        self.assertTrue(torch.allclose(x, x_denorm, atol=1e-5))

    def test_model_forward_pass_v11(self):
        """Test Standard HybridJointNetwork Forward Pass"""
        model = HybridJointNetwork(self.config)
        
        # Dummy Inputs
        B, T, D = 4, 60, 2
        S = 1 + 8 # 1 static col + 8 dim embedding
        x_dyn = torch.randn(B, T, D)
        x_stat = torch.randn(B, 1) # Only passing explicit static feats
        x_time = torch.zeros(B, 2, dtype=torch.long)
        
        # Override strict dimension checking in test by forcing config
        model.static_proj = torch.nn.Sequential(
            torch.nn.Linear(1 + 8, 32), torch.nn.GELU(), torch.nn.Dropout(0)
        )
        
        r1, c1, r2, c2 = model(x_dyn, x_stat, x_time)
        self.assertEqual(r1.shape, (B, 1))
        self.assertEqual(c1.shape, (B, 3))

    def test_model_forward_pass_v17(self):
        """Test Experimental V17 (BiLSTM + KAN + RevIN)"""
        cfg = self.config.copy()
        cfg['bidirectional'] = True
        cfg['use_revin'] = True
        cfg['use_kan'] = True
        cfg['input_dim'] = 5
        cfg['static_dim'] = 1
        
        model = ExperimentalNetwork(cfg)
        
        B, T, D = 4, 60, 5
        x_dyn = torch.randn(B, T, D)
        x_stat = torch.randn(B, 1)
        x_time = torch.zeros(B, 2, dtype=torch.long)
        
        r1, c1, r2, c2 = model(x_dyn, x_stat, x_time)
        
        self.assertEqual(r1.shape, (B, 1))
        self.assertEqual(c1.shape, (B, 3))
        
    def test_training_loop(self):
        """Test ModelTrainer running one step"""
        model = HybridJointNetwork(self.config)
        trainer = ModelTrainer(model, self.config)
        
        # Create Dummy Loader
        # X_d, X_s, X_t, Y_1_reg, Y_1_cls, Y_2_reg, Y_2_cls
        X_d = torch.randn(10, 60, 2)
        X_s = torch.randn(10, 1)
        X_t = torch.zeros(10, 2, dtype=torch.long)
        Y_1r = torch.randn(10)
        Y_1c = torch.randint(0, 3, (10,))
        Y_2r = torch.randn(10)
        Y_2c = torch.randint(0, 3, (10,))
        
        dataset = torch.utils.data.TensorDataset(X_d, X_s, X_t, Y_1r, Y_1c, Y_2r, Y_2c)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        loss = trainer.train_epoch(loader)
        self.assertIsInstance(loss, float)
        self.assertFalse(np.isnan(loss))

if __name__ == '__main__':
    unittest.main()
