#!/usr/bin/env python3
"""Test script to verify Ollama integration in chat.py"""

import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    import ollama
    print("✓ Ollama library imported successfully")
    
    # Test connection to Ollama
    try:
        models = ollama.list()
        print(f"✓ Connected to Ollama successfully")
        print(f"  Available models: {[m['name'] for m in models.get('models', [])]}")
    except Exception as e:
        print(f"✗ Failed to connect to Ollama: {e}")
        print("  Make sure Ollama is running with: ollama serve")
        sys.exit(1)
    
    # Test a simple generation
    try:
        test_prompt = "Say 'Hello from Ollama!' in exactly 5 words."
        response = ollama.generate(
            model='llama3.2',  # Using default from config
            prompt=test_prompt,
            options={'num_predict': 10}
        )
        print(f"✓ Generation test successful")
        print(f"  Response: {response.get('response', '').strip()}")
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        print("  You may need to pull the model first: ollama pull llama3.2")
    
except ImportError:
    print("✗ Failed to import ollama library")
    print("  Install it with: pip install ollama")
    sys.exit(1)

print("\nOllama integration test complete!")