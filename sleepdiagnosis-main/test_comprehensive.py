#!/usr/bin/env python3
"""
Comprehensive test script for SleepApnea project
Tests backend API and checks for common frontend issues
"""

import requests
import json
import time
import os
import sys
from pathlib import Path

def test_backend_health():
    """Test backend health endpoint"""
    print("🔍 Testing backend health...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend health OK - Status: {data.get('status')}, Model loaded: {data.get('model_loaded')}")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend (http://localhost:8000)")
        return False
    except Exception as e:
        print(f"❌ Backend health check error: {e}")
        return False

def test_backend_root():
    """Test backend root endpoint"""
    print("🔍 Testing backend root endpoint...")
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend root OK - Message: {data.get('message')}")
            return True
        else:
            print(f"❌ Backend root failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend root error: {e}")
        return False

def test_frontend_accessibility():
    """Test frontend accessibility"""
    print("🔍 Testing frontend accessibility...")
    try:
        response = requests.get("http://localhost:9002/", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is accessible")
            return True
        else:
            print(f"❌ Frontend not accessible: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to frontend (http://localhost:9002)")
        return False
    except Exception as e:
        print(f"❌ Frontend accessibility error: {e}")
        return False

def test_audio_analysis_api():
    """Test audio analysis API with a simple test file"""
    print("🔍 Testing audio analysis API...")
    
    # Check if test audio file exists
    test_file = "c:/Users/Akarsh PR/Downloads/SleepApnea/backend/test_snore.wav"
    if not os.path.exists(test_file):
        print(f"⚠️  Test audio file not found: {test_file}")
        return False
    
    try:
        with open(test_file, 'rb') as f:
            files = {'audio': ('test_snore.wav', f, 'audio/wav')}
            response = requests.post("http://localhost:8000/analyze", files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Audio analysis OK - Label: {data.get('label')}, Confidence: {data.get('confidence_score', 0)}%")
            
            # Check if spectrogram was generated
            if data.get('spectrogram'):
                print("✅ Spectrogram data included in response")
            else:
                print("⚠️  No spectrogram data in response")
            
            return True
        else:
            print(f"❌ Audio analysis failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error detail: {error_data.get('detail', 'No detail')}")
            except:
                print(f"   Raw response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Audio analysis error: {e}")
        return False

def check_required_files():
    """Check if all required files exist"""
    print("🔍 Checking required files...")
    
    required_files = [
        "c:/Users/Akarsh PR/Downloads/SleepApnea/index.html",
        "c:/Users/Akarsh PR/Downloads/SleepApnea/backend/main.py",
        "c:/Users/Akarsh PR/Downloads/SleepApnea/backend/model.py",
        "c:/Users/Akarsh PR/Downloads/SleepApnea/backend/requirements.txt"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {os.path.basename(file_path)} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def check_backend_dependencies():
    """Check if backend Python dependencies are available"""
    print("🔍 Checking backend dependencies...")
    
    required_modules = [
        'fastapi',
        'uvicorn', 
        'librosa',
        'matplotlib',
        'sklearn',
        'numpy'
    ]
    
    import subprocess
    all_available = True
    
    for module in required_modules:
        try:
            result = subprocess.run([sys.executable, '-c', f'import {module}'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✅ {module} available")
            else:
                print(f"❌ {module} not available")
                all_available = False
        except Exception as e:
            print(f"❌ Error checking {module}: {e}")
            all_available = False
    
    return all_available

def main():
    """Run comprehensive tests"""
    print("SleepApnea Project - Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        ("Required Files", check_required_files),
        ("Backend Dependencies", check_backend_dependencies),
        ("Frontend Accessibility", test_frontend_accessibility),
        ("Backend Health", test_backend_health),
        ("Backend Root", test_backend_root),
        ("Audio Analysis API", test_audio_analysis_api)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        print("-" * 40)
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
    
    print("-" * 60)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The project appears to be working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above for details.")
        
        # Provide specific recommendations based on failures
        if not results.get("Backend Health"):
            print("\n💡 Backend Issues:")
            print("   - Make sure backend server is running: python backend/main.py")
            print("   - Check if port 8000 is available")
        
        if not results.get("Frontend Accessibility"):
            print("\n💡 Frontend Issues:")
            print("   - Make sure frontend server is running: python -m http.server 9002")
            print("   - Check if port 9002 is available")
        
        if not results.get("Audio Analysis API"):
            print("\n💡 API Issues:")
            print("   - Check backend logs for specific errors")
            print("   - Verify audio processing dependencies (librosa, ffmpeg)")

if __name__ == "__main__":
    main()