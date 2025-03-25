import requests

url = "http://127.0.0.1:8000/predict/"
data = {"features": [5.1, 3.5, 1.4, 0.2]}  # ✅ Dictionary format

response = requests.post(url, json=data)  # ✅ Use json=, NOT data=
print(response.json())  # Should return the prediction
