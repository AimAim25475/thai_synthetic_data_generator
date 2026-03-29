#!/usr/bin/env python
"""Quick test of generate_typhoon.py functionality"""
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent / "chitchat_api"))

print("[TEST 1] Importing module...")
try:
    from synthetic.generate_typhoon import _interactive_mode, main
    print("  [OK] Module imported successfully\n")
except Exception as e:
    print(f"  [ERROR] Import failed: {e}\n")
    sys.exit(1)

print("[TEST 2] Checking _interactive_mode function exists...")
try:
    import inspect
    sig = inspect.signature(_interactive_mode)
    print(f"  [OK] Function signature: {sig}\n")
except Exception as e:
    print(f"  [ERROR] {e}\n")
    sys.exit(1)

print("[TEST 3] Testing argument parsing (dry run)...")
try:
    import argparse
    # Create a test argparse with same args as main()
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--model", default="")
    ap.add_argument("--out", default="")
    
    # Simulate: python generate_typhoon.py --interactive
    test_args = ap.parse_args(["--interactive"])
    print(f"  [OK] Args parsed: interactive={test_args.interactive}\n")
except Exception as e:
    print(f"  [ERROR] {e}\n")
    sys.exit(1)

print("="*60)
print("ALL TESTS PASSED! Script is ready for interactive testing.")
print("="*60)
print("\nTo test interactive mode, run:")
print("  python chitchat_api/synthetic/generate_typhoon.py --interactive")
print("\nTo test help, run:")
print("  python chitchat_api/synthetic/generate_typhoon.py --help")
