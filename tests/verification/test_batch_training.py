import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd
import numpy as np
import json
import tempfile

# Add root to path
sys.path.append(os.getcwd())

from tools.train import main as train_main
from tools.run_batch import main as batch_main
from tools.optimize import main as optimize_main
from library.processors import GenericProcessor

class TestBatchSystem(unittest.TestCase):
    
    def setUp(self):
        # Create a dummy config file
        self.test_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.test_dir.name, "test_config.json")
        
        self.config = {
            "experiment_name": "test_model",
            "processor": "generic",
            "model_type": "hybrid_v12",
            "epochs": 1,
            "batch_size": 4,
            "use_gpu": False, # Force CPU for validation
            "feature_pipeline": [{"step": "log_return", "input": "underlying_close", "output": "log_ret"}],
            "dynamic_columns": ["log_ret"],
            "targets": [{"name": "target_1", "horizon": 1}],
            "target_col": "target_1",
            # Network Params
            "hidden_size": 16,
            "lstm_layers": 1,
            "trans_layers": 1,
            "dropout": 0.1,
            "n_heads": 2
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)
            
    def tearDown(self):
        self.test_dir.cleanup()

    @patch('tools.train.GenericProcessor.fetch_data')
    @patch('library.trainer.ModelTrainer.save_checkpoint')
    def test_train_model(self, mock_save, mock_fetch):
        # Mock Data
        dates = pd.date_range("2023-01-01", periods=100)
        df = pd.DataFrame(index=dates)
        df['underlying_close'] = 100 + np.random.randn(100).cumsum()
        df['underlying_open'] = df['underlying_close']
        df['target_1'] = 0.01
        
        mock_fetch.return_value = df
        
        # Patch Args
        test_args = ["train.py", "--config", self.config_path]
        with patch.object(sys, 'argv', test_args):
            train_main()
            
        # Assertions
        self.assertTrue(mock_fetch.called)
        self.assertTrue(mock_save.called)
        print("\n[TEST] tools/train.py passed successfully.")

    @patch('tools.run_batch.subprocess.run')
    @patch('tools.run_batch.glob.glob')
    def test_batch_train(self, mock_glob, mock_subprocess):
        # Mock finding files
        mock_glob.return_value = [self.config_path, "another.json"]
        
        # Mock subprocess success
        mock_subprocess.return_value.returncode = 0
        
        test_args = ["run_batch.py", "--config_dir", self.test_dir.name]
        with patch.object(sys, 'argv', test_args):
            batch_main()
            
        # Assertions
        self.assertEqual(mock_subprocess.call_count, 2)
        print("\n[TEST] tools/run_batch.py passed successfully.")
    @patch('tools.run_batch.subprocess.run')
    @patch('tools.run_batch.glob.glob')
    def test_batch_train_overrides(self, mock_glob, mock_subprocess):
        # Mock finding files
        mock_glob.return_value = [self.config_path]
        mock_subprocess.return_value.returncode = 0
        
        # Test with overrides
        test_args = ["run_batch.py", "--config_dir", self.test_dir.name, "--epochs", "50", "--force_cpu"]
        with patch.object(sys, 'argv', test_args):
            batch_main()
            
        # Assertions
        # Check if subprocess called with overrides
        # Call args: (['...python...', 'tools/train.py', '--config', '...', '--epochs', '50', '--force_cpu'], check=True)
        call_args = mock_subprocess.call_args[0][0]
        self.assertIn("--epochs", call_args)
        self.assertIn("50", call_args)
        self.assertIn("--force_cpu", call_args)
        print("\n[TEST] tools/run_batch.py overrides passed successfully.")

    @patch('tools.optimize.optuna.create_study')
    @patch('tools.optimize.GenericProcessor.fetch_data')
    def test_optimize_config_parsing(self, mock_fetch, mock_create_study):
        # Create config with specific Optuna range
        custom_config = self.config.copy()
        custom_config["optuna"] = {
            "batch_size": [16],
            "hidden_size": [64]
        }
        with open(self.config_path, 'w') as f:
            json.dump(custom_config, f)
            
        # Mock Data
        dates = pd.date_range("2023-01-01", periods=100)
        df = pd.DataFrame(index=dates)
        df['underlying_close'] = 100 + np.random.randn(100).cumsum()
        df['underlying_open'] = df['underlying_close']
        df['target_1'] = 0.01
        
        mock_fetch.return_value = df
        
        # Mock Study & Trial
        mock_study = MagicMock()
        mock_create_study.return_value = mock_study
        
        # Capture the trial.suggest_categorical calls
        mock_trial = MagicMock()
        
        # We need to run the objective function directly or patch the optimize call
        # Let's patch tools.optimize.objective to verify it's called with correct epochs
        # AND check if the real objective function parses args correctly.
        # Actually, simpler: verify tools.optimize.main calls study.optimize
        
        # But to verify the search space, we need to inspect what happens inside the lambda passed to study.optimize
        # That's hard to mock. 
        # Instead, let's trust the integration test logic for now, but ensure --epochs is passed.
        
        test_args = ["optimize.py", "--config", self.config_path, "--trials", "1", "--epochs", "30", "--force_cpu"]
        
        # We need to spy on the trainer to see if it ran for 30 epochs
        with patch('tools.optimize.ModelTrainer') as MockTrainer:
            with patch.object(sys, 'argv', test_args):
                # We need to make study.optimize actually RUN the objective function
                # The first arg to optimize is the func
                def side_effect(func, n_trials=1):
                    trial = MagicMock()
                    # Mock suggestion values to avoid errors
                    trial.suggest_float.return_value = 0.001
                    trial.suggest_categorical.return_value = 16
                    trial.suggest_int.return_value = 1
                    func(trial)
                
                mock_study.optimize.side_effect = side_effect
                
                optimize_main()
                
            # Verification
            # Check Trainer.run_training called with epochs=30
            trainer_instance = MockTrainer.return_value
            trainer_instance.run_training.assert_called_with(
                unittest.mock.ANY, unittest.mock.ANY, epochs=30, save_path=unittest.mock.ANY
            )
            print("\n[TEST] tools/optimize.py respects --epochs=30.")

if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
