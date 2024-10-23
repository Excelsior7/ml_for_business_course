import requests

url = "http://127.0.0.1:80/predict"  # URL de votre API locale

# Dictionnaire avec les paramètres à passer à l'API
data = {
    "remote_allowed": True,
    "work_type_contract": False,
    "work_type_full_time": True,
    "work_type_part_time": False,
    "state": "CL",
    "company_name": "TechCorp",
    "title": "Data Analyst",
    "description": "Analyze data and create reports"
}

# Envoyer la requête POST
response = requests.post(url, json=data)

# Afficher la réponse de l'API
if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print(f"Erreur: {response.status_code}")
