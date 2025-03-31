import requests
import json
import time

# --- Configuration ---
# Change this if your Flask app is running on a different address or port
BASE_URL = "http://localhost:5000"

HEALTH_ENDPOINT = f"{BASE_URL}/api/health"
RECOMMEND_ENDPOINT = f"{BASE_URL}/api/recommend"

# --- Test Functions ---

def test_health_check():
    """Tests the /api/health endpoint."""
    print("--- Testing /api/health ---")
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=10) # Add timeout

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            try:
                data = response.json()
                print("Response JSON:")
                print(json.dumps(data, indent=2))

                # Basic checks for expected keys
                assert 'status' in data and data['status'] == 'healthy'
                assert 'model' in data
                assert 'active_agents' in data
                assert 'ml_model_status' in data
                assert 'yield_model' in data['ml_model_status']
                assert 'sustainability_model' in data['ml_model_status']

                print("✅ Health Check Passed: Status is healthy and expected keys found.")
                return True
            except (json.JSONDecodeError, AssertionError) as e:
                print(f"❌ Health Check Failed: Error processing response or assertion failed: {e}")
                print(f"Raw Response Text: {response.text}")
                return False
        else:
            print(f"❌ Health Check Failed: Received non-200 status code.")
            print(f"Raw Response Text: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Health Check Failed: Could not connect to the server.")
        print(f"   Is the Flask app running at {BASE_URL}?")
        return False
    except requests.exceptions.Timeout:
         print("❌ Health Check Failed: Request timed out.")
         return False
    except Exception as e:
        print(f"❌ Health Check Failed: An unexpected error occurred: {e}")
        return False
    finally:
        print("-" * 27 + "\n")


def test_recommend_success():
    """Tests the /api/recommend endpoint with valid data."""
    print("--- Testing /api/recommend (Success Case) ---")
    payload = {
        "query": "What crops should I plant for good yield and sustainability?",
        "soil_ph": 6.7,
        "soil_moisture": 58.5,
        "temperature": 23.0,
        "rainfall": 195.0,
        "location": "Iowa, USA", # Optional
        "crop": "Corn"         # Optional (influences ML input if provided)
    }
    print("Sending Payload:")
    print(json.dumps(payload, indent=2))

    try:
        response = requests.post(RECOMMEND_ENDPOINT, json=payload, timeout=90) # Longer timeout for agent processing

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            try:
                data = response.json()
                print("Response JSON (Partial):")
                # Print selected keys for brevity
                print(f"  Final Recommendation (Exists): {'final_recommendation' in data and bool(data.get('final_recommendation'))}")
                print(f"  ML Yield Prediction: {data.get('ml_yield_prediction_tons_acre')}")
                print(f"  ML Sustainability Prediction: {data.get('ml_sustainability_score_prediction')}")
                print(f"  Recommended Crops (Heuristic): {data.get('recommended_crops_heuristic')}")
                print(f"  Agent Insights (Exists): {'agent_insights' in data}")
                # print(json.dumps(data, indent=2)) # Uncomment to see full response

                # Check for key structure elements
                assert 'final_recommendation' in data
                assert 'ml_yield_prediction_tons_acre' in data # Can be None if ML failed
                assert 'ml_sustainability_score_prediction' in data # Can be None if ML failed
                assert 'recommended_crops_heuristic' in data and isinstance(data['recommended_crops_heuristic'], list)
                assert 'sustainability_data_avg' in data and isinstance(data['sustainability_data_avg'], dict)
                assert 'market_data_avg' in data and isinstance(data['market_data_avg'], dict)
                assert 'agent_insights' in data and isinstance(data['agent_insights'], dict)
                assert 'farmer' in data['agent_insights']
                assert 'weather' in data['agent_insights']
                assert 'market' in data['agent_insights']

                print("✅ Recommend Success Test Passed: Received 200 and expected response structure.")
                return True

            except (json.JSONDecodeError, AssertionError, KeyError) as e:
                print(f"❌ Recommend Success Test Failed: Error processing response or assertion failed: {e}")
                print(f"Raw Response Text: {response.text[:500]}...") # Print start of text
                return False
        else:
            print(f"❌ Recommend Success Test Failed: Received non-200 status code.")
            print(f"Raw Response Text: {response.text[:500]}...")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Recommend Success Test Failed: Could not connect to the server.")
        return False
    except requests.exceptions.Timeout:
         print("❌ Recommend Success Test Failed: Request timed out. Agents might be taking too long.")
         return False
    except Exception as e:
        print(f"❌ Recommend Success Test Failed: An unexpected error occurred: {e}")
        return False
    finally:
        print("-" * 42 + "\n")


def test_recommend_missing_param():
    """Tests the /api/recommend endpoint with a missing required parameter."""
    print("--- Testing /api/recommend (Missing Parameter) ---")
    payload = {
        "query": "Testing missing rainfall",
        "soil_ph": 7.0,
        "soil_moisture": 50.0,
        "temperature": 25.0
        # "rainfall" is missing
    }
    print("Sending Payload:")
    print(json.dumps(payload, indent=2))

    try:
        response = requests.post(RECOMMEND_ENDPOINT, json=payload, timeout=15)

        print(f"Status Code: {response.status_code}")

        if response.status_code == 400:
            try:
                data = response.json()
                print("Response JSON:")
                print(json.dumps(data, indent=2))
                assert 'error' in data
                assert 'rainfall' in data.get('error', '').lower() # Check if 'rainfall' mentioned in error
                print("✅ Recommend Missing Param Test Passed: Received 400 and error message as expected.")
                return True
            except (json.JSONDecodeError, AssertionError, KeyError) as e:
                 print(f"❌ Recommend Missing Param Test Failed: Received 400, but response format/content unexpected: {e}")
                 print(f"Raw Response Text: {response.text}")
                 return False
        else:
            print(f"❌ Recommend Missing Param Test Failed: Expected 400, but received {response.status_code}.")
            print(f"Raw Response Text: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Recommend Missing Param Test Failed: Could not connect to the server.")
        return False
    except requests.exceptions.Timeout:
         print("❌ Recommend Missing Param Test Failed: Request timed out.")
         return False
    except Exception as e:
        print(f"❌ Recommend Missing Param Test Failed: An unexpected error occurred: {e}")
        return False
    finally:
        print("-" * 46 + "\n")


def test_recommend_invalid_param_type():
    """Tests the /api/recommend endpoint with an invalid parameter type."""
    print("--- Testing /api/recommend (Invalid Parameter Type) ---")
    payload = {
        "query": "Testing invalid type",
        "soil_ph": "high", # Invalid type, should be float
        "soil_moisture": 50.0,
        "temperature": 25.0,
        "rainfall": 150.0
    }
    print("Sending Payload:")
    print(json.dumps(payload, indent=2))

    try:
        response = requests.post(RECOMMEND_ENDPOINT, json=payload, timeout=15)

        print(f"Status Code: {response.status_code}")

        if response.status_code == 400:
            try:
                data = response.json()
                print("Response JSON:")
                print(json.dumps(data, indent=2))
                assert 'error' in data
                assert 'invalid' in data.get('error', '').lower() or 'non-numeric' in data.get('error', '').lower()
                assert 'soil_ph' in data.get('error', '') # Check if the correct param is mentioned
                print("✅ Recommend Invalid Param Type Test Passed: Received 400 and error message as expected.")
                return True
            except (json.JSONDecodeError, AssertionError, KeyError) as e:
                 print(f"❌ Recommend Invalid Param Type Test Failed: Received 400, but response format/content unexpected: {e}")
                 print(f"Raw Response Text: {response.text}")
                 return False
        else:
            print(f"❌ Recommend Invalid Param Type Test Failed: Expected 400, but received {response.status_code}.")
            print(f"Raw Response Text: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Recommend Invalid Param Type Test Failed: Could not connect to the server.")
        return False
    except requests.exceptions.Timeout:
         print("❌ Recommend Invalid Param Type Test Failed: Request timed out.")
         return False
    except Exception as e:
        print(f"❌ Recommend Invalid Param Type Test Failed: An unexpected error occurred: {e}")
        return False
    finally:
        print("-" * 50 + "\n")


# --- Run Tests ---
if __name__ == "__main__":
    print(f"Starting API tests against {BASE_URL}\n")
    results = {}
    results['health_check'] = test_health_check()
    time.sleep(1) # Small pause
    results['recommend_success'] = test_recommend_success()
    time.sleep(1)
    results['recommend_missing_param'] = test_recommend_missing_param()
    time.sleep(1)
    results['recommend_invalid_type'] = test_recommend_invalid_param_type()

    print("\n--- Test Summary ---")
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + ("=" * 18))
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed.")
    print(("=" * 18))