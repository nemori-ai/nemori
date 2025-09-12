#!/usr/bin/env python3
"""
Test script for LoCoMo data sampling validation.
This script demonstrates how to use the sampling feature to test with small amounts of data.
"""

import subprocess
import sys
import os
from pathlib import Path

def test_sampling():
    """Test the LoCoMo processing with data sampling."""
    
    print("🧪 Testing LoCoMo Processing with Data Sampling")
    print("=" * 60)
    
    # Change to the evaluation scripts directory
    script_dir = Path(__file__).parent / "evaluation" / "memos" / "evaluation" / "scripts" / "locomo"
    if not script_dir.exists():
        print(f"❌ Script directory not found: {script_dir}")
        return False
    
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Test configurations
    test_configs = [
        {
            "name": "Minimal Test (1 conversation)",
            "sample_size": 1,
            "max_concurrency": 1,
            "version": "test_sample_1"
        },
        {
            "name": "Small Test (3 conversations)", 
            "sample_size": 3,
            "max_concurrency": 2,
            "version": "test_sample_3"
        },
        {
            "name": "Medium Test (5 conversations)",
            "sample_size": 5, 
            "max_concurrency": 3,
            "version": "test_sample_5"
        }
    ]
    
    for config in test_configs:
        print(f"\n🚀 Running {config['name']}")
        print("-" * 40)
        
        cmd = [
            sys.executable, "locomo_ingestion_emb_full.py",
            "--lib", "nemori",
            "--version", config["version"],
            "--max-concurrency", str(config["max_concurrency"]),
            "--sample-size", str(config["sample_size"])
        ]
        
        print(f"📝 Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                print(f"✅ {config['name']} completed successfully")
                print(f"📊 Sample processing completed for {config['sample_size']} conversations")
                
                # Show some output highlights
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if any(keyword in line for keyword in ['✅ Finished processing', '📊 Complete Conversation Processing', '🎉 LoCoMo Complete Processing']):
                        print(f"   {line}")
                        
            else:
                print(f"❌ {config['name']} failed with return code: {result.returncode}")
                print(f"Error output: {result.stderr[-500:]}")  # Last 500 chars of error
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {config['name']} timed out after 5 minutes")
        except Exception as e:
            print(f"❌ {config['name']} failed with exception: {e}")
    
    print("\n🎯 Sampling Tests Completed!")
    print("=" * 60)
    print("📝 You can now run individual tests using:")
    print("   python locomo_ingestion_emb_full.py --sample-size 1 --version test_quick")
    print("   python locomo_ingestion_emb_full.py --sample-size 3 --version test_small") 
    print("   python locomo_ingestion_emb_full.py --sample-size 5 --version test_medium")
    print("\n💡 For full dataset processing (all 10 conversations):")
    print("   python locomo_ingestion_emb_full.py --version full_semantic_emb")

if __name__ == "__main__":
    test_sampling()