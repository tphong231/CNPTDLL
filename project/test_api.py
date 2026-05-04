"""
API Testing Examples

Test the FastAPI recommendation system endpoints.
Make sure the API is running before running these tests:
    python -m uvicorn api.main:app --reload
"""

import requests
import json
from typing import Dict, Any

# API configuration
API_BASE_URL = "http://localhost:8000/api/v1"
HEADERS = {"Content-Type": "application/json"}


class APITester:
    """Helper class for testing API endpoints."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
    
    def _request(self, method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Make API request."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, headers=HEADERS)
            elif method == "POST":
                response = requests.post(url, json=data, headers=HEADERS)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            print(f"\n{method} {endpoint}")
            print(f"Status: {response.status_code}")
            
            if response.text:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2, default=str)}")
                return result
            else:
                print("(No response body)")
                return {}
        
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return {}
    
    # Health & Status
    def test_health(self):
        """Test health endpoint."""
        print("\n" + "="*60)
        print("TEST: Health Check")
        print("="*60)
        return self._request("GET", "/health")
    
    def test_stats(self):
        """Test statistics endpoint."""
        print("\n" + "="*60)
        print("TEST: Get Statistics")
        print("="*60)
        return self._request("GET", "/stats")
    
    # Users
    def test_create_user(self, user_id: int, username: str = None, email: str = None):
        """Test user creation."""
        print("\n" + "="*60)
        print("TEST: Create User")
        print("="*60)
        
        data = {
            "user_id": user_id,
            "username": username or f"user_{user_id}",
            "email": email
        }
        
        return self._request("POST", "/users/create", data)
    
    def test_get_user(self, user_id: int):
        """Test get user."""
        print("\n" + "="*60)
        print("TEST: Get User")
        print("="*60)
        return self._request("GET", f"/users/{user_id}")
    
    # Ratings
    def test_submit_rating(self, user_id: int, movie_id: int, rating: float):
        """Test rating submission."""
        print("\n" + "="*60)
        print("TEST: Submit Rating")
        print("="*60)
        
        data = {
            "user_id": user_id,
            "movie_id": movie_id,
            "rating": rating
        }
        
        return self._request("POST", "/ratings", data)
    
    def test_get_user_ratings(self, user_id: int):
        """Test get user ratings."""
        print("\n" + "="*60)
        print("TEST: Get User Ratings")
        print("="*60)
        return self._request("GET", f"/users/{user_id}/ratings")
    
    def test_get_movie_ratings(self, movie_id: int):
        """Test get movie ratings."""
        print("\n" + "="*60)
        print("TEST: Get Movie Ratings")
        print("="*60)
        return self._request("GET", f"/movies/{movie_id}/ratings")
    
    # Recommendations
    def test_get_recommendations(self, user_id: int, n_recommendations: int = 5, 
                                model_name: str = "hybrid"):
        """Test get recommendations."""
        print("\n" + "="*60)
        print("TEST: Get Recommendations")
        print("="*60)
        
        data = {
            "user_id": user_id,
            "n_recommendations": n_recommendations,
            "model_name": model_name
        }
        
        return self._request("POST", "/recommend", data)
    
    def test_get_stored_recommendations(self, user_id: int, model_name: str = "hybrid"):
        """Test get stored recommendations."""
        print("\n" + "="*60)
        print("TEST: Get Stored Recommendations")
        print("="*60)
        return self._request("GET", f"/users/{user_id}/recommendations/{model_name}")
    
    # Predictions
    def test_predict_rating(self, user_id: int, movie_id: int):
        """Test rating prediction."""
        print("\n" + "="*60)
        print("TEST: Predict Rating")
        print("="*60)
        
        data = {
            "user_id": user_id,
            "movie_id": movie_id
        }
        
        return self._request("POST", "/predict", data)
    
    def test_list_models(self):
        """Test list available models."""
        print("\n" + "="*60)
        print("TEST: List Available Models")
        print("="*60)
        return self._request("GET", "/models")


def run_full_test_suite():
    """Run complete API test suite."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  RECOMMENDATION SYSTEM - API TESTS".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    tester = APITester()
    
    try:
        # Health check
        tester.test_health()
        
        # Statistics
        tester.test_stats()
        
        # List models
        tester.test_list_models()
        
        # Create user
        tester.test_create_user(user_id=1, username="alice", email="alice@example.com")
        
        # Get user
        tester.test_get_user(user_id=1)
        
        # Submit ratings
        tester.test_submit_rating(user_id=1, movie_id=1, rating=5)
        tester.test_submit_rating(user_id=1, movie_id=2, rating=4)
        tester.test_submit_rating(user_id=1, movie_id=3, rating=3)
        
        # Get user ratings
        tester.test_get_user_ratings(user_id=1)
        
        # Get movie ratings
        tester.test_get_movie_ratings(movie_id=1)
        
        # Get recommendations
        tester.test_get_recommendations(user_id=1, n_recommendations=5, model_name="hybrid")
        tester.test_get_recommendations(user_id=1, n_recommendations=5, model_name="content_based")
        tester.test_get_recommendations(user_id=1, n_recommendations=5, model_name="collaborative")
        
        # Get stored recommendations
        tester.test_get_stored_recommendations(user_id=1, model_name="hybrid")
        
        # Predict rating
        tester.test_predict_rating(user_id=1, movie_id=10)
        
        # Create another user
        tester.test_create_user(user_id=2, username="bob", email="bob@example.com")
        
        # Submit ratings for user 2
        tester.test_submit_rating(user_id=2, movie_id=1, rating=3)
        tester.test_submit_rating(user_id=2, movie_id=2, rating=4)
        
        # Get recommendations for user 2
        tester.test_get_recommendations(user_id=2, n_recommendations=3, model_name="hybrid")
        
        # Final stats
        tester.test_stats()
        
        print("\n" + "="*60)
        print("✓ All API tests completed!")
        print("="*60 + "\n")
    
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


def run_minimal_test():
    """Run minimal test to verify API is working."""
    print("\nRunning minimal API test...")
    
    tester = APITester()
    
    try:
        result = tester.test_health()
        
        if result.get("status") == "healthy":
            print("\n✓ API is healthy and working!")
            return True
        else:
            print("\n✗ API returned unexpected response")
            return False
    
    except Exception as e:
        print(f"\n✗ API connection failed: {str(e)}")
        print("Make sure the API is running: python -m uvicorn api.main:app --reload")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--minimal":
        run_minimal_test()
    else:
        run_full_test_suite()
