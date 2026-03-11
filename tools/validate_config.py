import os
import sys
import json
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

CRITICAL_RULES = {
    # If using V23 MultiRes architecture, it MUST have at least 2 transformer layers
    "min_trans_layers": 2, 
    # Minimum training epochs for a real run
    "min_epochs": 10,
    # Capacity locks for Titan Scale
    "titan_hidden_size": 1024,
    "titan_batch_size": 32
}

def validate_config(config_path, enforce_titan=False):
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return False
        
    with open(config_path, 'r') as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in {config_path}")
            return False

    errors = []
    warnings = []

    # 1. Epoch Check (Prevent 1-epoch regression)
    epochs = config.get("epochs", 0)
    if epochs < CRITICAL_RULES["min_epochs"]:
        errors.append(f"CRITICAL: epochs ({epochs}) is less than minimum required ({CRITICAL_RULES['min_epochs']}) for a real run.")

    # 2. Transformer Layer Check (Prevent rollback to pure LSTM)
    model_type = config.get("model_type", "")
    if "v23" in model_type.lower() or "multires" in model_type.lower():
        trans_layers = config.get("trans_layers", 0)
        if trans_layers < CRITICAL_RULES["min_trans_layers"]:
            errors.append(f"CRITICAL: V23 architecture requires at least {CRITICAL_RULES['min_trans_layers']} trans_layers. Found {trans_layers}.")

    # 3. Titan Capacity Lock (Optional enforcement)
    if enforce_titan:
        hs = config.get("hidden_size", 0)
        bs = config.get("batch_size", 0)
        if hs != CRITICAL_RULES["titan_hidden_size"]:
            errors.append(f"TITAN LOCK VIOLATION: hidden_size must be {CRITICAL_RULES['titan_hidden_size']}. Found {hs}.")
        if bs != CRITICAL_RULES["titan_batch_size"]:
            errors.append(f"TITAN LOCK VIOLATION: batch_size must be {CRITICAL_RULES['titan_batch_size']}. Found {bs}.")
    else:
        # Just warn if under capacity
        hs = config.get("hidden_size", 0)
        if hs < 256:
            warnings.append(f"Warning: Low hidden_size ({hs}). Network might lack capacity.")

    # Report
    if errors:
        logger.error(f"[X] Validation FAILED for {config_path}:")
        for e in errors:
            logger.error(f"  - {e}")
    else:
        logger.info(f"[+] Validation PASSED for {config_path}.")
        
    for w in warnings:
         logger.warning(f"  - {w}")

    return len(errors) == 0

def main():
    parser = argparse.ArgumentParser(description="V23 Config Validaton Tool")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    parser.add_argument("--enforce_titan", action="store_true", help="Strictly enforce Titan capacity (hidden_size 1024, batch_size 32)")
    args = parser.parse_args()
    
    success = validate_config(args.config, args.enforce_titan)
    if not success:
        sys.exit(1)
        
if __name__ == "__main__":
    main()
