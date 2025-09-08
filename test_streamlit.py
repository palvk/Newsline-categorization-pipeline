#!/usr/bin/env python3
"""
Test streamlit app functionality
"""

import pandas as pd
import sys
import os

def test_streamlit_imports():
    """Test if streamlit app can import required modules"""
    print("Testing streamlit imports...")
    
    try:
        import streamlit as st
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        from final_mkt_analyzer import HybridEnricher, enrich_dataset
        print("SUCCESS: All required imports available")
        return True
    except ImportError as e:
        print(f"FAILED: Import error - {e}")
        return False

def test_data_processing():
    """Test data processing functionality"""
    print("Testing data processing...")
    
    try:
        from final_mkt_analyzer import enrich_dataset
        
        # Create sample data
        sample_df = pd.DataFrame({
            'headlines': ['Test Article'],
            'content': ['Interest rates are rising, affecting borrowing costs for businesses.'],
            'url': [''],
            'source': ['Test']
        })
        
        # Process data
        enriched_df = enrich_dataset(sample_df)
        
        # Check required columns exist
        required_cols = ['sector', 'impact', 'reason']
        missing_cols = [col for col in required_cols if col not in enriched_df.columns]
        
        if missing_cols:
            print(f"FAILED: Missing columns: {missing_cols}")
            return False
        
        print("SUCCESS: Data processing works")
        print(f"  Columns: {list(enriched_df.columns)}")
        return True
        
    except Exception as e:
        print(f"FAILED: Data processing error - {e}")
        return False

def test_business_data():
    """Test processing of business_data.csv"""
    print("Testing business data processing...")
    
    try:
        if not os.path.exists("business_data.csv"):
            print("WARNING: business_data.csv not found")
            return True
        
        df = pd.read_csv("business_data.csv")
        print(f"  Loaded {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")
        
        from final_mkt_analyzer import enrich_dataset
        enriched_df = enrich_dataset(df.head(5))  # Test with first 5 rows
        
        print(f"  Enriched columns: {list(enriched_df.columns)}")
        print("SUCCESS: Business data processing works")
        return True
        
    except Exception as e:
        print(f"FAILED: Business data processing error - {e}")
        return False

def main():
    """Run all tests"""
    print("STREAMLIT APP TEST")
    print("=" * 30)
    
    tests = [
        test_streamlit_imports,
        test_data_processing,
        test_business_data
    ]
    
    passed = 0
    for test in tests:
        print()
        if test():
            passed += 1
    
    print("\n" + "=" * 30)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\nSUCCESS: Streamlit app should work!")
        print("Run: streamlit run streamlit_simple.py")
    else:
        print(f"\nWARNING: {len(tests) - passed} tests failed")
    
    return passed == len(tests)

if __name__ == "__main__":
    main()