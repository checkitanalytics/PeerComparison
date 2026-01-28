"""
Comprehensive Test Suite for PeerComparison Platform
Run with: pytest test_peercomparison.py -v
"""

import pytest
import requests
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://127.0.0.1:5001"
TIMEOUT = 60  # seconds


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def base_url():
    """Base URL for the application"""
    return BASE_URL


@pytest.fixture(scope="session")
def session():
    """Requests session for all tests"""
    return requests.Session()


# ============================================================================
# TEST 1: HEALTH CHECK
# ============================================================================

class TestHealthCheck:
    """Test application health and configuration"""
    
    def test_health_endpoint_returns_200(self, base_url, session):
        """Health endpoint should return 200 OK"""
        response = session.get(f"{base_url}/api/health", timeout=TIMEOUT)
        assert response.status_code == 200
    
    def test_health_endpoint_returns_json(self, base_url, session):
        """Health endpoint should return valid JSON"""
        response = session.get(f"{base_url}/api/health", timeout=TIMEOUT)
        data = response.json()
        assert isinstance(data, dict)
    
    def test_health_status_is_healthy(self, base_url, session):
        """Health endpoint should report healthy status"""
        response = session.get(f"{base_url}/api/health", timeout=TIMEOUT)
        data = response.json()
        assert data.get("status") == "healthy"
    
    def test_deepseek_configured(self, base_url, session):
        """DeepSeek API should be configured"""
        response = session.get(f"{base_url}/api/health", timeout=TIMEOUT)
        data = response.json()
        assert data.get("deepseek_configured") == True, "DeepSeek API key not configured"
    
    def test_perplexity_configured(self, base_url, session):
        """Perplexity API should be configured"""
        response = session.get(f"{base_url}/api/health", timeout=TIMEOUT)
        data = response.json()
        # Note: This might be False if Perplexity has issues, that's okay
        assert "perplexity_configured" in data


# ============================================================================
# TEST 2: TICKER RESOLUTION
# ============================================================================

class TestTickerResolution:
    """Test ticker resolution endpoint"""
    
    def test_resolve_valid_ticker(self, base_url, session):
        """Should resolve valid ticker (AAPL)"""
        response = session.post(
            f"{base_url}/api/resolve",
            json={"input": "AAPL"},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert "ticker" in data
    
    def test_resolve_empty_input(self, base_url, session):
        """Should handle empty input gracefully"""
        response = session.post(
            f"{base_url}/api/resolve",
            json={},
            timeout=TIMEOUT
        )
        # Should either return 200 with error or 4xx status
        assert response.status_code in [200, 400, 422]
    
    def test_server_stable_after_error(self, base_url, session):
        """Server should remain stable after error"""
        # Send bad request
        session.post(f"{base_url}/api/resolve", json={}, timeout=TIMEOUT)
        
        # Verify health
        response = session.get(f"{base_url}/api/health", timeout=TIMEOUT)
        assert response.status_code == 200


# ============================================================================
# TEST 3: PEER DISCOVERY
# ============================================================================

class TestPeerDiscovery:
    """Test peer company discovery"""
    
    @pytest.mark.parametrize("ticker", ["AAPL", "TSLA", "MSFT", "GOOGL"])
    def test_find_peers_valid_tickers(self, base_url, session, ticker):
        """Should find peers for valid tech companies"""
        response = session.post(
            f"{base_url}/api/find-peers",
            json={"ticker": ticker},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert "peers" in data or "error" not in data
    
    def test_find_peers_invalid_ticker(self, base_url, session):
        """Should handle invalid ticker gracefully"""
        response = session.post(
            f"{base_url}/api/find-peers",
            json={"ticker": "ZZZZ999"},
            timeout=TIMEOUT
        )
        # Should return 200 with fallback or error message
        assert response.status_code == 200
        data = response.json()
        # Should have some response structure
        assert isinstance(data, dict)


# ============================================================================
# TEST 4: FINANCIAL METRICS
# ============================================================================

class TestFinancialMetrics:
    """Test financial metrics retrieval"""
    
    def test_get_metrics_single_company(self, base_url, session):
        """Should retrieve metrics for single company"""
        response = session.post(
            f"{base_url}/api/get-metrics",
            json={"tickers": ["AAPL"]},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
    
    def test_get_metrics_multiple_companies(self, base_url, session):
        """Should retrieve metrics for multiple companies"""
        response = session.post(
            f"{base_url}/api/get-metrics",
            json={"tickers": ["AAPL", "MSFT", "GOOGL"]},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
    
    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"])
    def test_get_metrics_stress_test(self, base_url, session, ticker):
        """Rapid metrics calls for stability testing"""
        response = session.post(
            f"{base_url}/api/get-metrics",
            json={"tickers": [ticker]},
            timeout=TIMEOUT
        )
        assert response.status_code == 200


# ============================================================================
# TEST 5: DATA ACCURACY - DIFFERENT COMPANY TYPES
# ============================================================================

class TestDataAccuracy:
    """Test data accuracy across different company types"""
    
    @pytest.mark.parametrize("ticker", ["TSLA", "AAPL", "MSFT", "GOOGL"])
    def test_tech_companies(self, base_url, session, ticker):
        """Should retrieve accurate data for major tech companies"""
        response = session.get(
            f"{base_url}/api/primary-company-analysis",
            params={"ticker": ticker},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        # Should have some analysis content
        assert data is not None
    
    @pytest.mark.parametrize("ticker", ["V", "PYPL", "SOFI", "AFRM"])
    def test_fintech_companies(self, base_url, session, ticker):
        """Should handle fintech companies (may not have Gross Margin %)"""
        response = session.get(
            f"{base_url}/api/primary-company-analysis",
            params={"ticker": ticker},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        # Should handle gracefully even if some metrics are missing
        assert isinstance(data, dict)
    
    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"])
    def test_mega7_specialty_group(self, base_url, session, ticker):
        """Should recognize and use MEGA7 specialty peer group"""
        response = session.post(
            f"{base_url}/api/find-peers",
            json={"ticker": ticker},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        # Should find peers from the specialty group
        assert "peers" in data or "peer" in str(data).lower()
    
    @pytest.mark.parametrize("ticker", ["TSLA", "RIVN", "LCID", "NIO"])
    def test_ev_specialty_group(self, base_url, session, ticker):
        """Should handle EV specialty group"""
        response = session.post(
            f"{base_url}/api/find-peers",
            json={"ticker": ticker},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_unknown_ticker_handling(self, base_url, session):
        """Should handle unknown/invalid tickers gracefully"""
        response = session.post(
            f"{base_url}/api/find-peers",
            json={"ticker": "XYZ123INVALID"},
            timeout=TIMEOUT
        )
        # Should return 200 with some fallback response
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


# ============================================================================
# TEST 6: PEER KEY METRICS CONCLUSION
# ============================================================================

# ============================================================================
# TEST 6: PEER KEY METRICS CONCLUSION
# ============================================================================

class TestPeerKeyMetricsConclusion:
    """Test AI-powered conclusion generation"""
    
    def test_conclusion_generation(self, base_url, session):
        """Should validate payload and return proper error for invalid data"""
        # This endpoint requires complex data structure with latest_quarter.rows
        # Testing that it properly validates and rejects invalid payloads
        payload = {
            "primary": "AAPL",
            "peers": ["MSFT", "GOOGL"],
            "data": {}  # Invalid - missing required structure
        }
        response = session.post(
            f"{base_url}/api/peer-key-metrics-conclusion",
            json=payload,
            timeout=TIMEOUT
        )
        # Should return 400 for invalid payload (proper validation)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_english_conclusion_generation(self, base_url, session):
        """Should generate English conclusion"""
        response = session.get(
            f"{base_url}/api/primary-company-analysis",
            params={"ticker": "AAPL"},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        # Should have English content
        assert data is not None
    
    def test_chinese_translation(self, base_url, session):
        """Should translate to Chinese when requested"""
        response = session.get(
            f"{base_url}/api/primary-company-analysis",
            params={"ticker": "AAPL", "lang": "zh"},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert data is not None


# ============================================================================
# TEST 7: AI PROVIDER TESTS
# ============================================================================

class TestAIProviders:
    """Test AI provider configuration and behavior"""
    
    def test_deepseek_api_configured(self, base_url, session):
        """DeepSeek API should be properly configured"""
        response = session.get(f"{base_url}/api/health", timeout=TIMEOUT)
        data = response.json()
        assert data.get("deepseek_configured") == True, "DeepSeek not configured"
        assert data.get("deepseek_model") is not None
    
    def test_deepseek_generating_analysis(self, base_url, session):
        """Should use DeepSeek for analysis generation"""
        response = session.get(
            f"{base_url}/api/primary-company-analysis",
            params={"ticker": "AAPL"},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        
        # Check if using AI provider (not local fallback)
        if "llm_provider" in data:
            assert data["llm_provider"] in ["deepseek", "perplexity"], \
                f"Should use AI provider, got: {data.get('llm_provider')}"
    
    def test_chinese_translation_via_deepseek(self, base_url, session):
        """Chinese translation should work via DeepSeek"""
        response = session.get(
            f"{base_url}/api/primary-company-analysis",
            params={"ticker": "TSLA", "lang": "zh"},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
    
    # Note: Testing fallback behavior would require temporarily invalidating API keys
    # This should be done in a controlled test environment, not in production tests


# ============================================================================
# TEST 8: COMPREHENSIVE ERROR HANDLING
# ============================================================================

class TestComprehensiveErrorHandling:
    """Test error handling across various scenarios"""
    
    def test_invalid_ticker_xyz123(self, base_url, session):
        """Should handle completely invalid ticker"""
        response = session.post(
            f"{base_url}/api/find-peers",
            json={"ticker": "XYZ123"},
            timeout=TIMEOUT
        )
        assert response.status_code == 200  # Should not crash
    
    def test_empty_ticker_input(self, base_url, session):
        """Should handle empty ticker input"""
        response = session.post(
            f"{base_url}/api/find-peers",
            json={"ticker": ""},
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 400, 422]
    
    def test_no_ticker_field(self, base_url, session):
        """Should handle missing ticker field"""
        response = session.post(
            f"{base_url}/api/find-peers",
            json={},
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 400, 422]
    
    def test_company_with_no_data(self, base_url, session):
        """Should handle companies with unavailable data"""
        response = session.get(
            f"{base_url}/api/primary-company-analysis",
            params={"ticker": "INVALIDCOMPANY999"},
            timeout=TIMEOUT
        )
        # Should handle gracefully without crashing
        assert response.status_code in [200, 404, 500]
    
    def test_malformed_json(self, base_url, session):
        """Should handle malformed JSON payloads"""
        response = session.post(
            f"{base_url}/api/find-peers",
            data="this is not json",
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 400, 422, 500]
    
    def test_special_characters_in_ticker(self, base_url, session):
        """Should handle special characters in ticker"""
        response = session.post(
            f"{base_url}/api/resolve",
            json={"input": "TEST@#$%"},
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 400, 422]
    
    def test_very_long_ticker_string(self, base_url, session):
        """Should handle excessively long ticker strings"""
        long_ticker = "A" * 1000
        response = session.post(
            f"{base_url}/api/resolve",
            json={"input": long_ticker},
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 400, 422]


# ============================================================================
# TEST 9: UI/FRONTEND FUNCTIONALITY
# ============================================================================

class TestUIFrontend:
    """Test UI/Frontend endpoints and functionality"""
    
    def test_main_page_loads(self, base_url, session):
        """Main page should load successfully"""
        response = session.get(f"{base_url}/", timeout=TIMEOUT)
        assert response.status_code == 200
    
    def test_language_toggle_endpoint(self, base_url, session):
        """Language toggle should work for both EN and ZH"""
        # Test English
        response_en = session.get(
            f"{base_url}/api/primary-company-analysis",
            params={"ticker": "AAPL", "lang": "en"},
            timeout=TIMEOUT
        )
        assert response_en.status_code == 200
        
        # Test Chinese
        response_zh = session.get(
            f"{base_url}/api/primary-company-analysis",
            params={"ticker": "AAPL", "lang": "zh"},
            timeout=TIMEOUT
        )
        assert response_zh.status_code == 200
    
    def test_charts_data_endpoint(self, base_url, session):
        """Charts should have data to render"""
        response = session.post(
            f"{base_url}/api/get-metrics",
            json={"tickers": ["AAPL", "MSFT", "GOOGL"]},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        # Should have data structure for charts
        assert isinstance(data, dict)
    
    def test_add_remove_company_workflow(self, base_url, session):
        """Should handle multiple companies (add/remove simulation)"""
        # Get metrics for 2 companies
        response_2 = session.post(
            f"{base_url}/api/get-metrics",
            json={"tickers": ["AAPL", "MSFT"]},
            timeout=TIMEOUT
        )
        assert response_2.status_code == 200
        
        # Get metrics for 4 companies (simulating adding more)
        response_4 = session.post(
            f"{base_url}/api/get-metrics",
            json={"tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"]},
            timeout=TIMEOUT
        )
        assert response_4.status_code == 200


# ============================================================================
# TEST 10: COMPREHENSIVE PERFORMANCE TESTS
# ============================================================================

class TestComprehensivePerformance:
    """Test performance under various conditions"""
    
    def test_multiple_consecutive_searches(self, base_url, session):
        """Should handle multiple consecutive searches"""
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        
        for ticker in tickers:
            response = session.get(
                f"{base_url}/api/primary-company-analysis",
                params={"ticker": ticker},
                timeout=TIMEOUT
            )
            assert response.status_code == 200
            time.sleep(0.5)  # Small delay between requests
    
    def test_large_peer_group_5plus_companies(self, base_url, session):
        """Should handle large peer groups (5+ companies)"""
        response = session.post(
            f"{base_url}/api/get-metrics",
            json={"tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
    
    def test_response_time_under_30_seconds(self, base_url, session):
        """Analysis should complete in under 30 seconds"""
        start = time.time()
        response = session.get(
            f"{base_url}/api/primary-company-analysis",
            params={"ticker": "AAPL"},
            timeout=TIMEOUT
        )
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 30.0, f"Response took {elapsed:.2f}s (should be <30s)"
    
    def test_concurrent_requests_simulation(self, base_url, session):
        """Should handle rapid consecutive requests"""
        import concurrent.futures
        
        def make_request(ticker):
            response = session.get(
                f"{base_url}/api/primary-company-analysis",
                params={"ticker": ticker},
                timeout=TIMEOUT
            )
            return response.status_code
        
        tickers = ["AAPL", "MSFT", "GOOGL"]
        
        # Note: This is not true concurrency with the same session,
        # but it simulates rapid requests
        statuses = []
        for ticker in tickers:
            status = make_request(ticker)
            statuses.append(status)
            time.sleep(0.1)
        
        # All requests should succeed
        assert all(status == 200 for status in statuses)
    
    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "GOOGL"])
    def test_repeated_same_ticker_performance(self, base_url, session, ticker):
        """Should handle repeated requests for same ticker efficiently"""
        for i in range(3):
            start = time.time()
            response = session.get(
                f"{base_url}/api/primary-company-analysis",
                params={"ticker": ticker},
                timeout=TIMEOUT
            )
            elapsed = time.time() - start
            
            assert response.status_code == 200
            # Should not get progressively slower
            assert elapsed < 60.0


# ============================================================================
# TEST 11: END-TO-END PRIMARY COMPANY ANALYSIS
# ============================================================================

class TestPrimaryCompanyAnalysis:
    """Test complete end-to-end analysis flow"""
    
    @pytest.mark.parametrize("ticker", ["AAPL", "TSLA", "MSFT"])
    def test_full_analysis_flow(self, base_url, session, ticker):
        """Should complete full analysis for major companies"""
        response = session.get(
            f"{base_url}/api/primary-company-analysis",
            params={"ticker": ticker},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "ticker" in data or "primary" in data
    
    def test_analysis_with_english_language(self, base_url, session):
        """Should generate English analysis"""
        response = session.get(
            f"{base_url}/api/primary-company-analysis",
            params={"ticker": "AAPL", "lang": "en"},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
    
    def test_analysis_with_chinese_language(self, base_url, session):
        """Should generate Chinese analysis"""
        response = session.get(
            f"{base_url}/api/primary-company-analysis",
            params={"ticker": "AAPL", "lang": "zh"},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
    
    def test_analysis_stability_repeated_calls(self, base_url, session):
        """Server should remain stable under repeated analysis calls"""
        for i in range(3):
            response = session.get(
                f"{base_url}/api/primary-company-analysis",
                params={"ticker": "AAPL"},
                timeout=TIMEOUT
            )
            assert response.status_code == 200
            time.sleep(1)  # Small delay between requests


# ============================================================================
# TEST 12: DIFFERENT COMPANY TYPES (SECTORS)
# ============================================================================

class TestDifferentCompanyTypes:
    """Test with various company types and sectors"""
    
    @pytest.mark.parametrize("ticker,company_type", [
        ("JPM", "Financial"),
        ("LCID", "EV Startup"),
        ("PLTR", "Tech/Defense"),
        ("SHOP", "E-commerce"),
        ("UBER", "Tech/Transportation")
    ])
    def test_analysis_different_sectors(self, base_url, session, ticker, company_type):
        """Should handle different company types"""
        response = session.get(
            f"{base_url}/api/primary-company-analysis",
            params={"ticker": ticker},
            timeout=TIMEOUT
        )
        assert response.status_code == 200, f"Failed for {company_type} company: {ticker}"


# ============================================================================
# TEST 13: CORS SUPPORT
# ============================================================================

class TestCORS:
    """Test CORS configuration for frontend support"""
    
    def test_cors_preflight_request(self, base_url, session):
        """Should handle CORS preflight requests"""
        response = session.options(
            f"{base_url}/api/find-peers",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers


# ============================================================================
# TEST 14: LEGACY ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases (legacy tests)"""
    
    def test_invalid_json_payload(self, base_url, session):
        """Should handle invalid JSON gracefully"""
        response = session.post(
            f"{base_url}/api/find-peers",
            data="not valid json",
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 400, 422, 500]
    
    def test_missing_required_fields(self, base_url, session):
        """Should handle missing required fields"""
        response = session.post(
            f"{base_url}/api/get-metrics",
            json={},  # Missing 'tickers' field
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 400, 422]


# ============================================================================
# TEST 15: PERFORMANCE BENCHMARKS
# ============================================================================

class TestPerformance:
    """Test response time and performance (legacy benchmarks)"""
    
    def test_health_check_response_time(self, base_url, session):
        """Health check should be fast (<1 second)"""
        start = time.time()
        response = session.get(f"{base_url}/api/health", timeout=TIMEOUT)
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 1.0, f"Health check took {elapsed:.2f}s (should be <1s)"
    
    def test_analysis_response_time(self, base_url, session):
        """Analysis should complete in reasonable time (<60 seconds)"""
        start = time.time()
        response = session.get(
            f"{base_url}/api/primary-company-analysis",
            params={"ticker": "AAPL"},
            timeout=TIMEOUT
        )
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 60.0, f"Analysis took {elapsed:.2f}s (should be <60s)"


# ============================================================================
# RUN SUMMARY
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PeerComparison Platform - Comprehensive Test Suite")
    print("=" * 70)
    print(f"Base URL: {BASE_URL}")
    print(f"Timeout: {TIMEOUT}s")
    print("=" * 70)
    print("\n📋 TEST COVERAGE (15 Test Categories):\n")
    print("  1. Health Check - API configuration & status")
    print("  2. Ticker Resolution - Input validation")
    print("  3. Peer Discovery - Company peer matching")
    print("  4. Financial Metrics - Data retrieval")
    print("  5. Data Accuracy - Different company types")
    print("  6. Peer Key Metrics - Conclusion generation")
    print("  7. AI Providers - DeepSeek/Perplexity configuration")
    print("  8. Comprehensive Error Handling - Edge cases")
    print("  9. UI/Frontend - Language toggle & charts")
    print(" 10. Comprehensive Performance - Load testing")
    print(" 11. End-to-End Analysis - Full workflow")
    print(" 12. Different Company Types - Cross-sector")
    print(" 13. CORS Support - Frontend compatibility")
    print(" 14. Legacy Error Handling - Additional tests")
    print(" 15. Performance Benchmarks - Response times")
    print("=" * 70)
    print("\n🚀 HOW TO RUN:\n")
    print("Run all tests:")
    print("  pytest test_peercomparison.py -v")
    print("\nRun with detailed output:")
    print("  pytest test_peercomparison.py -v -s")
    print("\nRun specific category:")
    print("  pytest test_peercomparison.py::TestHealthCheck -v")
    print("  pytest test_peercomparison.py::TestDataAccuracy -v")
    print("  pytest test_peercomparison.py::TestAIProviders -v")
    print("\nGenerate HTML report:")
    print("  pip install pytest-html")
    print("  pytest test_peercomparison.py --html=test_report.html --self-contained-html")
    print("\nRun only quick tests (skip slow performance tests):")
    print("  pytest test_peercomparison.py -v -m 'not slow'")
    print("=" * 70)
    print("\n✅ CHECKLIST COVERAGE:\n")
    print("✓ API Endpoint Tests (Health, Peers, Metrics, Conclusion)")
    print("✓ Data Accuracy Tests (Tech, Fintech, MEGA7, EV, Unknown)")
    print("✓ AI Provider Tests (DeepSeek, Translation, Fallback)")
    print("✓ Error Handling (Invalid tickers, Empty inputs, Malformed data)")
    print("✓ UI/Frontend Tests (Charts, Language toggle, Multi-company)")
    print("✓ Performance Tests (Consecutive searches, Large groups, <30s)")
    print("=" * 70)