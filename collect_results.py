import os
import re
import json
import argparse
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple, Set


def split_shot(dataset_name: str) -> Tuple[str, int]:
    """Split 'DTD8shot' -> ('DTD', 8); full-shot names -> (name, 0)."""
    m = re.search(r'(\d+)shot', dataset_name)
    if m:
        return dataset_name[:m.start()] + dataset_name[m.end():], int(m.group(1))
    return dataset_name, 0


def find_completed_experiments(input_dir: str) -> List[str]:
    """Find directories containing completed experiments"""
    completed_dirs = []
    for root, _, files in os.walk(input_dir):
        if 'results.json' in files:
            completed_dirs.append(root)
    return completed_dirs

def read_experiment_data(exp_dir: str) -> Tuple[Dict, float]:
    """Read experiment parameters and results from a directory"""
    # Read args.json
    with open(os.path.join(exp_dir, 'config.json'), 'r') as f:
        args = json.load(f)
    
    # Read test_acc
    with open(os.path.join(exp_dir, 'results.json'), 'r') as f:
        try:
            results = json.load(f)
        except:
            print(exp_dir)
            exit(0)
    test_acc = results['test_acc']
    
    return args, test_acc

def get_unique_values(all_args: List[Dict], key: str) -> Set[str]:
    """Extract unique values for a given key from all experiments"""
    return set(args[key] for args in all_args)

class MethodKey:
    """Class to handle method key creation and formatting"""
    def __init__(self, args: Dict):
        self.epochs = args.get('epochs', 0)
        self.batch_size = args.get('batch_size', 0)
        self.lr = args.get('lr', 0)
        # Drop the dataset directory from the caption path (structure:
        # .../<captioner>/<dataset>/<file>) so the same method over different
        # datasets collapses into one row (dataset is a column, not a row split).
        self.captions = self._norm_captions(args.get('captions'))
        self.add_class_template = args.get('add_class_template')
        self.w = args.get('w')
        self.prompt_ensemble = args.get('prompt_ensemble', False)

        # Determine method name from args
        if self.captions in ('class_label', None):
            self.method_name = 'FLYP(ens)' if self.prompt_ensemble else 'FLYP'
        else:
            self.method_name = 'MMFT'

        self.method_args = {
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'captions': self.captions,
            'add_class_template': self.add_class_template,
            'w': self.w,
            'prompt_ensemble': self.prompt_ensemble if self.prompt_ensemble else None,
        }
        self.method_args = {k: v for k, v in self.method_args.items() if v is not None}

    @staticmethod
    def _norm_captions(captions):
        """Remove the dataset directory so caption paths are dataset-agnostic."""
        if not captions or captions == 'class_label':
            return captions
        parts = captions.replace('\\', '/').split('/')
        if len(parts) >= 2:
            del parts[-2]  # the dataset dir sits between captioner and filename
        return '/'.join(parts)

    def get_main_config(self) -> str:
        """Get main method configuration (method and prompt size)"""
        return f"{self.method_name}"

    def get_hyper_params(self) -> str:
        """Get hyperparameter configuration"""
        # Sort method args for consistent formatting
        sorted_method_args = dict(sorted(self.method_args.items()))
        method_args_str = ', '.join([f"{k}={v}" for k, v in sorted_method_args.items()]) \
                          if sorted_method_args else 'None'
        
        return f"{method_args_str}"

    def get_sorting_key(self) -> Tuple:
        """Create a sorting key for consistent ordering"""
        # Sort method args for consistent sorting
        sorted_method_args = sorted(self.method_args.items())
        return (
            self.method_name, 
            self.lr,
            self.epochs, 
            self.batch_size, 
            sorted_method_args
        )

    def __hash__(self):
        return hash((self.method_name, self.epochs, 
                    self.batch_size, self.lr, 
                    tuple(sorted(self.method_args.items()))))

    def __eq__(self, other):
        if not isinstance(other, MethodKey):
            return False
        return (self.method_name == other.method_name and 
                self.epochs == other.epochs and
                self.batch_size == other.batch_size and
                self.method_args == other.method_args)

def collect_results(args) -> Tuple[List[str], List[str], Dict]:
    """Collect and organize experiment results"""
    completed_dirs = find_completed_experiments(args.input_dir)
    print(f"Found {len(completed_dirs)} completed experiments")
    
    # Read all experiment data
    all_data = []
    for exp_dir in completed_dirs:
        exp_args, test_acc = read_experiment_data(exp_dir)
        all_data.append((exp_args, test_acc))
    
    # Get unique models
    all_args = [exp_args for exp_args, _ in all_data]
    models = sorted(get_unique_values(all_args, 'model_name'))

    # Organize results by model, shot, method, and base dataset
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    datasets_by_shot = defaultdict(set)  # shot -> {base dataset}
    for exp_args, test_acc in all_data:
        model = exp_args['model_name']
        base, shot = split_shot(exp_args['dataset'])
        method_key = MethodKey(exp_args)
        results[model][shot][method_key][base].append(test_acc)
        datasets_by_shot[shot].add(base)

    return datasets_by_shot, models, results, len(completed_dirs)

def format_mean_std_error(values: List[float]) -> str:
    """Format mean and standard error in LaTeX format"""
    if not values:
        return '-'
    mean = np.mean(values)
    std_error = np.std(values) / np.sqrt(len(values))
    if len(values) < 3:
        return f"*{mean:.2f} \\fontsize{{5pt}}{{5pt}}\\selectfont{{$\\pm$ {std_error:.2f}}}"
    else:
        return f"{mean:.2f} \\fontsize{{5pt}}{{5pt}}\\selectfont{{$\\pm$ {std_error:.2f}}}"

def generate_latex_table(model: str, datasets: List[str],
                        model_results: Dict, shot: int = 0) -> str:
    """Generate LaTeX table for a specific model (and shot setting)"""
    # Calculate table width based on number of datasets
    table_width = len(datasets) + 3  # method + hyperparams + datasets + average columns

    # Start table
    latex = f"\\begin{{table}}[h]\n"
    latex += "\\centering\n\\small\n"
    latex += "\\begin{adjustbox}{max width=\\textwidth}\n"
    latex += f"\\begin{{tabular}}{{ll{'c' * (table_width-2)}}}\n"
    latex += "\\toprule\n"
    
    # Header row 
    latex += "Method & Hyperparameters & " + \
             " & ".join(datasets).replace('_', '\_') + " & Average \\\\\n"
    latex += "\\midrule\n"
    
    # Group results by main configuration
    grouped_results = defaultdict(list)
    for method_key in model_results.keys():
        main_config = method_key.get_main_config()
        grouped_results[main_config].append(method_key)
   
    # Results for each method group
    first_in_group = True
    for main_config, method_keys in sorted(grouped_results.items()):
        # Sort method keys using the new sorting key method
        sorted_method_keys = sorted(method_keys, key=lambda x: str(x.get_sorting_key()))
        
        for method_key in sorted_method_keys:
            dataset_results = model_results[method_key]
            row_values = []
            dataset_means = []
            
            # Collect results for each dataset
            for dataset in datasets:
                values = dataset_results.get(dataset, [])
                if values:
                    dataset_means.append(np.mean(values))
                formatted = format_mean_std_error(values)
                row_values.append(formatted)
            
            # Calculate average across datasets
            if dataset_means:
                overall_mean = np.mean(dataset_means)
                # overall_std = np.std(dataset_means) / np.sqrt(len(dataset_means))
                # overall_result = f"{overall_mean:.2f} $\\pm$ {overall_std:.2f}"
                overall_result = f"{overall_mean:.2f}"
            else:
                overall_result = "-"
            
            # Add row to table
            if first_in_group:
                main_method = main_config
                first_in_group = False
            else:
                main_method = "\\quad"  # Indent for subsequent rows
            
            row = f"{main_method} & {method_key.get_hyper_params()} & " + \
                    f"{' & '.join(row_values)} & {overall_result} \\\\\n"
            row = row.replace("_", "\\_")
            latex += row
        
        # Add small space between groups
        if not first_in_group:
            latex += "\\addlinespace[2pt]\n"
            first_in_group = True
    
    # Table footer
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{adjustbox}\n"
    shot_str = 'full-shot' if shot == 0 else f'{shot}-shot'
    latex += f"\\caption{{Results for {model} ({shot_str}). * indicates fewer than three experiments.}}\n".replace("_", "\\_")
    latex += f"\\label{{tab:results_{model}_{shot}shot}}\n"
    latex += "\\end{table}\n\n"
    
    return latex

def main():
    parser = argparse.ArgumentParser(description='Collect and summarize experiment results')
    parser.add_argument('--dir', dest='input_dir', default='output',
                       help='Directory containing experiment results')
    args = parser.parse_args()

   
    # Collect results
    datasets_by_shot, models, results, num_results = collect_results(args)
    output_path = os.path.join(args.input_dir, 'results.tex')

    # Generate LaTeX document and save to file
    with open(output_path, 'w') as f:
        # Document header
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{adjustbox}\n")
        f.write("\\usepackage[margin=20truemm]{geometry}\n")
        f.write("\\begin{document}\n\n")
        f.write(f"Found {num_results} completed experiments\n")
        
        # Generate tables for each model, split by shot setting
        for model in models:
            f.write(f"\\section{{{model}}}\n\n".replace("_", "\\_"))
            for shot in sorted(results[model].keys()):
                base_datasets = sorted(datasets_by_shot[shot])
                shot_str = 'full-shot' if shot == 0 else f'{shot}-shot'
                f.write(f"\\subsection*{{{shot_str}}}\n\n")
                latex_table = generate_latex_table(
                    model, base_datasets, results[model][shot], shot)
                f.write(latex_table)
        
        # Document footer
        f.write("\\end{document}\n")
    
    print(f"Results have been saved to: {output_path}")

if __name__ == "__main__":
    main()
