import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict


def create_experiment_hash(args: argparse.Namespace) -> str:
    """Generate hash from experiment settings"""
    args_str = json.dumps(vars(args), sort_keys=True)
    return hashlib.md5(args_str.encode()).hexdigest()[:8]


def setup_experiment_dir(args: argparse.Namespace) -> Path:
    """Create experiment directory and check if experiment is already completed"""
    exp_hash = create_experiment_hash(args)
    exp_dir = Path(args.output_dir) / exp_hash
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if experiment is already completed
    best_model_path = exp_dir / 'best_model.pth'
    results_path = exp_dir / 'results.json'
    
    if best_model_path.exists() and results_path.exists():
        print(f"Experiment already completed in directory: {exp_dir}")
        print("Both best_model.pth and results.json exist.")
        print("Skipping experiment execution.")
        exit(0)
    
    # Save experiment settings
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    return exp_dir


def load_experiment_config(exp_dir: Path) -> Dict:
    """Load experiment configuration from config.json"""
    config_path = exp_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded experiment config from: {config_path}")
    return config