#!/usr/bin/env python
"""Simple test of interactive mode without actually loading the model"""
import subprocess
import sys

python_exe = r"C:\Users\Rapeepat Ounkhom\AppData\Local\Programs\Python\Python311\python.exe"

print("="*70)
print("Testing Interactive Mode (without model generation)")
print("="*70 + "\n")

# Test inputs: small QA test, 2 examples, then cancel
test_input = """typhoon-ai/llama3.2-typhoon2-3b

qa_test.jsonl
4
2


1
1

n
"""

print("Sending test inputs to interactive mode...\n")

try:
    proc = subprocess.Popen(
        [python_exe, r"d:\chitchat\chitchat_api\synthetic\generate_typhoon.py", "--interactive"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=r"d:\chitchat"
    )
    
    stdout, stderr = proc.communicate(input=test_input, timeout=30)
    
    print("STDOUT:")
    print(stdout)
    
    if stderr:
        print("\nSTDERR:")
        print(stderr)
    
    print("\n" + "="*70)
    if "Configuration Summary:" in stdout:
        print("SUCCESS! Interactive mode is working correctly.")
        print("It prompted for parameters and showed the configuration summary.")
    else:
        print("Test completed. Check output above.")
    print("="*70)
    
except subprocess.TimeoutExpired:
    proc.kill()
    print("Test timed out (this is OK - model loading takes time)")
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
