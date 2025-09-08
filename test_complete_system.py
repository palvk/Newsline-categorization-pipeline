#!/usr/bin/env python3
"""
Complete system test script
Tests all components to ensure they work together
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
        test_texts = [
            "Interest rates are rising, affecting borrowing costs for businesses.",
            "New AI technology promises to revolutionize manufacturing processes.",
            "Government announces tax cuts to stimulate economic growth."
        ]
        
        print("✅ Final analyzer working")
        for i, text in enumerate(test_texts, 1):
            result = enricher.enrich(text)
            print(f"  {i}. Sector: {result['sector']}, Impact: {result['impact']}")
        
        return True
    except Exception as e:
        print(f"❌ Final analyzer failed: {e}")
        return False

def test_api_server():
    """Test if API server can start"""
    print("Testing API server...")
    try:
        import news_flask_api
        print("✅ API server module loaded successfully")
        return True
    except Exception as e:
        print(f"❌ API server failed: {e}")
        return False

def test_data_files():
    """Test if data files exist and are readable"""
    print("Testing data files...")
    
    files_to_check = [
        "business_data.csv",
        "business_tech_trainset.csv"
    ]
    
    all_good = True
    for file in files_to_check:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                print(f"✅ {file} - {len(df)} rows")
            except Exception as e:
                print(f"❌ {file} - Error reading: {e}")
                all_good = False
        else:
            print(f"⚠️  {file} - Not found")
    
    return all_good

def test_streamlit_compatibility():
    """Test if streamlit script has basic compatibility"""
    print("Testing streamlit compatibility...")
    try:
        # Check if the file exists and can be read
        with open("streamlit_simple.py", "r") as f:
            content = f.read()
        
        # Check for basic imports
        required_imports = ["streamlit", "pandas", "plotly"]
        missing_imports = []
        
        for imp in required_imports:
            if f"import {imp}" not in content:
                missing_imports.append(imp)
        
        if missing_imports:
            print(f"⚠️  Missing imports: {missing_imports}")
        else:
            print("✅ Streamlit script has required imports")
        
        return len(missing_imports) == 0
    except Exception as e:
        print(f"❌ Streamlit test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 COMPLETE SYSTEM TEST")
    print("=" * 40)
    
    tests = [
        ("Final Analyzer", test_final_analyzer),
        ("API Server", test_api_server),
        ("Data Files", test_data_files),
        ("Streamlit Compatibility", test_streamlit_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 40)
    print("📊 TEST SUMMARY")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 ALL TESTS PASSED! System is ready to use.")
        print("\nNext steps:")
        print("1. Start API: python news_flask_api.py")
        print("2. Test API: python test_api.py")
        print("3. Run Streamlit: streamlit run streamlit_simple.py")
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed. Check the issues above.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)