#!/usr/bin/env python3
"""
Simple system test without Unicode characters
"""

import sys
import os
import pandas as pd

def test_final_analyzer():
    """Test the final market analyzer"""
    print("Testing final market analyzer...")
    try:
        from final_mkt_analyzer import HybridEnricher
        
        enricher = HybridEnricher()
        test_text = "Interest rates are rising, affecting borrowing costs for businesses."
        result = enricher.enrich(test_text)
        
        print("SUCCESS: Final analyzer working")
        print(f"  Sector: {result['sector']}")
        print(f"  Impact: {result['impact']}")
        print(f"  Reason: {result['reason']}")
        
        return True
    except Exception as e:
        print(f"FAILED: Final analyzer - {e}")
        return False

def test_data_files():
    """Test if data files exist"""
    print("Testing data files...")
    
    files = ["business_data.csv", "business_tech_trainset.csv"]
    all_good = True
    
    for file in files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                print(f"SUCCESS: {file} - {len(df)} rows")
            except Exception as e:
                print(f"FAILED: {file} - {e}")
                all_good = False
        else:
            print(f"WARNING: {file} - Not found")
    
    return all_good

def test_api_module():
    """Test if API module can be imported"""
    print("Testing API module...")
    try:
        import news_flask_api
        print("SUCCESS: API module imported")
        return True
    except Exception as e:
        print(f"FAILED: API module - {e}")
        return False

def main():
    """Run tests"""
    print("COMPLETE SYSTEM TEST")
    print("=" * 30)
    
    tests = [
        test_final_analyzer,
        test_data_files,
        test_api_module
    ]
    
    passed = 0
    for test in tests:
        print()
        if test():
            passed += 1
    
    print("\n" + "=" * 30)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\nSUCCESS: All tests passed!")
        print("System is ready to use.")
    else:
        print(f"\nWARNING: {len(tests) - passed} tests failed")
    
    return passed == len(tests)

if __name__ == "__main__":
    main()