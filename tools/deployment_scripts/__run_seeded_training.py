import subprocess
import sys
import random

configs = [
    r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training\weights\V23_Deploy\v23_alpha_generation_rank5_PROTECTED_CAPACITY_BRUTE_58PERCENT_2026_03_04.json"
]

# Generate 50 distinct random seeds to deeply scan the loss basin
random.seed(4242) 
seeds = random.sample(range(10000, 99999), 50)

for config in configs:
    for seed in seeds:
        config_name = config.split('\\')[-1].replace('.json', '')
        print(f"\n========================================================")
        print(f"Running config: {config_name}")
        print(f"PyTorch Initializer Seed: {seed}")
        print(f"Expected Output: weights/{config_name}_seed_{seed}.pth")
        print(f"========================================================")
        
        cmd = [
            sys.executable, "tools/train.py",
            "--config", config,
            "--seed", str(seed)
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"Failed on seed {seed} for {config_name}. Continuing to next seed...")

print("\nAll 50 seeded training runs (1 config x 50 seeds) are complete!")
