import os
import subprocess
import sys

def run_script(script_path):
    print(f"\n{'='*50}\nRunning {script_path}...\n{'='*50}")
    result = subprocess.run([sys.executable, script_path], cwd=os.path.dirname(script_path))
    if result.returncode != 0:
        print(f"ERROR: {script_path} failed with exit code {result.returncode}")
        sys.exit(1)

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    scripts = [
        os.path.join(data_dir, 'ctg', 'preprocess_ctg.py'),
        os.path.join(data_dir, 'mice', 'preprocess_mice.py'),
        os.path.join(data_dir, 'diabetes', 'preprocess_diabetes.py'),
        os.path.join(data_dir, 'scrna', 'preprocess_scrna.py'),
        os.path.join(data_dir, 'tcga', 'preprocess_brca.py'),
        os.path.join(data_dir, 'tcga', 'update_brca_staging.py'),
        os.path.join(data_dir, 'tcga_gbmlgg', 'preprocess_gbmlgg.py')
    ]
    
    print("This script will download and preprocess all required datasets for the BTC benchmarks.")
    print("Standard benchmark datasets (e.g., Dermatology, Vehicle, Thyroid) are fetched on-the-fly via OpenML during runtime.")
    print("This script prepares the complex medical and biological datasets.\n")
    
    for script in scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"Warning: Could not find script {script}")
            
    print("\nAll medical datasets have been successfully downloaded and preprocessed!")
    print("You are now ready to run reproduce.ps1 or reproduce.sh.")

if __name__ == '__main__':
    main()
