#!/usr/bin/env python3
"""
Simple smoke test script for FastAPI endpoints.
Run this script to verify the main endpoints are working.
"""

import json
import sys
import time
from typing import Any, Dict

import requests


def test_endpoint(
    url: str,
    method: str = "GET",
    data: Dict[Any, Any] = None,
    headers: Dict[str, str] = None,
) -> bool:
    """Test a single endpoint and return success status."""
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=10)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=10)
        else:
            print(f"❌ Unsupported method: {method}")
            return False

        print(f"✅ {method} {url} - Status: {response.status_code}")

        # Print response for health endpoints
        if "/health" in url:
            try:
                response_data = response.json()
                print(f"   Response: {json.dumps(response_data, indent=2)}")
            except:
                print(f"   Response: {response.text}")

        return response.status_code < 400

    except requests.exceptions.RequestException as e:
        print(f"❌ {method} {url} - Error: {e}")
        return False


def main():
    """Run smoke tests for all key endpoints."""
    base_url = "http://127.0.0.1:8000"

    print("🚀 Starting FastAPI smoke tests...")
    print(f"Base URL: {base_url}")
    print("=" * 50)

    # Test endpoints
    endpoints = [
        # Health checks
        ("GET", "/", "Root endpoint"),
        ("GET", "/health", "Health check"),
        ("GET", "/health/readiness", "Readiness probe"),
        ("GET", "/health/liveness", "Liveness probe"),
        # API documentation
        ("GET", "/docs", "API documentation"),
        ("GET", "/openapi.json", "OpenAPI schema"),
        # Sample forecast endpoint (should require auth but still return 401/403)
        ("GET", "/forecast?store_id=STORE001&sku=SKU001", "Forecast endpoint"),
    ]

    results = []
    for method, endpoint, description in endpoints:
        print(f"\n🔍 Testing {description}...")
        url = f"{base_url}{endpoint}"
        success = test_endpoint(url, method)
        results.append((description, success))

        # Small delay between requests
        time.sleep(0.5)

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {description}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The API is working correctly.")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the application.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
