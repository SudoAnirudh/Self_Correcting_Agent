import os
import time
import requests

def test_nvidia(model):
    api_key = "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"
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Write a one-sentence greeting."}
        ],
        "temperature": 0.0,
        "max_tokens": 50
    }
    
    print(f"Calling NVIDIA model '{model}'...")
    start = time.time()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=40)
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()['choices'][0]['message']['content']}")
        print(f"Duration: {time.time() - start:.2f} seconds")
    except Exception as e:
        print(f"Failed with exception: {e}")
        print(f"Duration: {time.time() - start:.2f} seconds")

print("--- Testing 8B model ---")
test_nvidia("meta/llama-3.1-8b-instruct")

print("\n--- Testing 70B model ---")
test_nvidia("meta/llama-3.1-70b-instruct")
