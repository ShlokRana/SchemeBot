import argparse
import subprocess
import os
import sys

def train_models():
    """Train both entity recognition and intent classification models."""
    # Get the absolute path to the project root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set PYTHONPATH environment variable to include the project root
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{root_dir}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = root_dir
    
    print("Training entity recognition model...")
    subprocess.run(["python", "src/entity_extraction.py"], check=True, env=env)
    
    print("\nTraining intent classification model...")
    subprocess.run(["python", "src/train_nlu.py"], check=True, env=env)

def run_app():
    """Run the Streamlit application."""
    # Get the absolute path to the project root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set PYTHONPATH environment variable to include the project root
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{root_dir}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = root_dir
    
    subprocess.run(["streamlit", "run", "app.py"], check=True, env=env)

def main():
    parser = argparse.ArgumentParser(description="SchemeBot - Government Scheme Assistant")
    parser.add_argument("--train", action="store_true", help="Train the models")
    parser.add_argument("--run", action="store_true", help="Run the Streamlit app")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    if args.train:
        train_models()
    elif args.run:
        run_app()
    else:
        print("Please specify either --train to train models or --run to start the application")
        print("Example usage:")
        print("  python main.py --train  # Train the models")
        print("  python main.py --run    # Run the application")
        print("  python main.py --train --debug  # Train with verbose output")

if __name__ == "__main__":
    main()
