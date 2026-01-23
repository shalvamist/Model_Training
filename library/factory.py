import logging
import json
import os
import torch
from .processors import ProcessorV11, ProcessorV13, ProcessorV15
from .network_architectures import HybridJointNetwork, ExperimentalNetwork

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory to instantiate models and processors.
    """
    
    @staticmethod
    def get_processor(version, config):
        if version == 'v11': return ProcessorV11(config)
        if version == 'v13': return ProcessorV13(config)
        if version == 'v15': return ProcessorV15(config)
        raise ValueError(f"Unknown Processor Version: {version}")
        
    @staticmethod
    def create_model(config):
        # Check if experimental
        if config.get('model_type') == 'experimental_v17':
            logger.info("Instantiating Experimental V17 Network (RevIN + BiLSTM + KAN)")
            return ExperimentalNetwork(config)
            
        return HybridJointNetwork(config)
        
    @staticmethod
    def load_model_from_weights(config_path, weights_path, device='cpu'):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
            
        model = ModelFactory.create_model(cfg)
        
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                logger.warning(f"Key mismatch during load. Attempting strict=False. Error: {e}")
                model.load_state_dict(state_dict, strict=False)
                
        model.to(device)
        model.eval()
        return model
