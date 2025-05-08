#!/usr/bin/env python3
"""
Diagnostic script to check environment configuration.
"""

import os
import sys
import pathlib

def print_env_diagnostics():
    """
    Print detailed environment diagnostics.
    """
    print("Python Environment Diagnostics:")
    print("=" * 50)
    
    # Python path
    print("\nPython Path:")
    for path in sys.path:
        print(f"  {path}")
    
    # Current working directory
    print("\nCurrent Working Directory:")
    print(f"  {os.getcwd()}")
    
    # Script location
    print("\nScript Location:")
    print(f"  {pathlib.Path(__file__).resolve()}")
    
    # Environment variables
    print("\nEnvironment Variables:")
    for key, value in os.environ.items():
        print(f"  {key}: {value}")
    
    # .env file check
    print("\n.env File Check:")
    project_root = pathlib.Path(__file__).parent.parent
    env_file = project_root / '.env'
    
    print(f"  Checking {env_file}")
    print(f"  File exists: {env_file.exists()}")
    if env_file.exists():
        print(f"  File readable: {os.access(env_file, os.R_OK)}")
        print("  File contents:")
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('OPENAI_API_KEY='):
                        print(f"    {line.strip()[:20]}...{line.strip()[-20:]}")
                    else:
                        print(f"    {line.strip()}")
        except Exception as e:
            print(f"  Error reading file: {e}")

def main():
    """
    Main function to run environment diagnostics.
    """
    print_env_diagnostics()

if __name__ == "__main__":
    main()
