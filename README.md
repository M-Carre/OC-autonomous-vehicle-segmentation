# Projet de Segmentation d'Images pour Véhicule Autonome (OpenClassrooms - Projet 8)

Ce projet vise à développer un système de segmentation sémantique d'images pour des scènes de conduite urbaine, typiquement issues de caméras embarquées sur des véhicules autonomes. L'objectif est d'identifier et de délimiter différentes classes d'objets (routes, véhicules, piétons, bâtiments, etc.) dans une image. Le projet comprend l'entraînement d'un modèle de deep learning, la création d'une API pour servir ce modèle, une application web pour la démonstration, et le déploiement de l'API sur Azure via Docker et CI/CD avec GitHub Actions.

## 🌟 Fonctionnalités

*   **Modèle de Segmentation d'Images :** Entraînement d'un modèle U-Net (et potentiellement d'autres architectures comme ResNet50-UNet) sur le dataset Cityscapes pour segmenter les images en 8 catégories principales.
*   **API de Prédiction :** Une API FastAPI qui prend une image en entrée et retourne le masque de segmentation prédit en couleur.
*   **Application Web de Démonstration :** Une application Streamlit qui permet de téléverser une image ou de choisir parmi des exemples pour tester l'API de segmentation et visualiser les résultats.
*   **Conteneurisation :** L'API FastAPI est conteneurisée avec Docker pour faciliter le déploiement et assurer la reproductibilité.
*   **Déploiement Cloud :** L'API conteneurisée est déployée sur Azure App Service. L'application Streamlit peut également être déployée.
*   **CI/CD :** Un workflow GitHub Actions automatise les tests unitaires, la construction de l'image Docker, le push vers Azure Container Registry (ACR), et le déploiement sur Azure App Service à chaque modification poussée sur la branche principale.

## 📂 Structure du Dépôt

```
.
├── .github/workflows/        # Workflows GitHub Actions (CI/CD)
│   └── deploy_api_to_azure.yml
├── Dockerfile                # Dockerfile pour l'API FastAPI (si à la racine)
├── api/                      # Code source de l'API FastAPI
│   ├── main.py
│   ├── requirements_api.txt
│   └── tests/                # Tests unitaires pour l'API
│       └── test_main_api.py
├── models/                   # Modèles Keras entraînés (ex: .keras)
│   └── UNetMini_DevSet_NoAug_AggressiveCB_best.keras
├── notebooks/                # Notebooks Jupyter pour l'exploration et l'entraînement des modèles
│   └── notebook_modélisation_p8.ipynb
├── webapp/                   # Application web Streamlit
│   ├── app_streamlit.py
│   ├── requirements_streamlit.txt (si séparé)
│   └── sample_test_images/   # Images d'exemple pour l'app Streamlit
├── src/                      # Code source partagé (ex: utilitaires, mapping des classes)
│   └── cityscapes_labels.py
├── .dockerignore
├── .gitignore
└── README.md
```

## 🛠️ Technologies et Outils Utilisés

*   **Langage :** Python 3.10+
*   **Deep Learning :** TensorFlow, Keras
*   **Manipulation d'Images :** OpenCV, Pillow, Albumentations (pour l'augmentation)
*   **API Backend :** FastAPI, Uvicorn, Gunicorn
*   **Application Frontend :** Streamlit
*   **Conteneurisation :** Docker
*   **Cloud & Déploiement :** Azure (App Service, Azure Container Registry)
*   **CI/CD :** GitHub Actions
*   **Tests :** Pytest
*   **Suivi d'Expériences :** MLflow (utilisé dans les notebooks)
*   **Dataset :** Cityscapes (sous-ensemble pour le développement)

## 🚀 Mise en Route (Développement Local)

### Prérequis

*   Python 3.10 ou supérieur
*   pip (gestionnaire de paquets Python)
*   Docker Desktop (pour exécuter l'API conteneurisée localement)
*   Un environnement virtuel Python est fortement recommandé.

### 1. Cloner le Dépôt

```bash
git clone https://github.com/M-Carre/OC-autonomous-vehicle-segmentation.git
cd OC-autonomous-vehicle-segmentation
```

### 2. Environnement Virtuel et Dépendances

Il est recommandé de créer des environnements virtuels séparés pour l'API et l'application web, ou un environnement commun.

**Pour l'API FastAPI :**
```bash
# (Optionnel) Créer et activer un environnement virtuel
# python -m venv .venv_api
# source .venv_api/bin/activate  # Sur Linux/macOS
# .\.venv_api\Scripts\activate   # Sur Windows

pip install -r api/requirements_api.txt
```

**Pour l'Application Streamlit :**
```bash
# (Optionnel) Créer et activer un autre environnement virtuel
# python -m venv .venv_webapp
# source .venv_webapp/bin/activate
# .\.venv_webapp\Scripts\activate

pip install -r webapp/requirements_streamlit.txt # Si vous avez un fichier séparé
# Ou pip install streamlit requests Pillow si les dépendances sont minimales
```

### 3. Lancer l'API FastAPI Localement (avec Uvicorn, hors Docker)

Assurez-vous que le modèle (`.keras`) est présent dans le dossier `models/`.
```bash
cd api
uvicorn main:app --reload
```
L'API sera accessible sur `http://127.0.0.1:8000`. Les docs interactives (Swagger UI) sur `http://127.0.0.1:8000/docs`.

### 4. Lancer l'API FastAPI Localement (avec Docker)

Assurez-vous que Docker Desktop est en cours d'exécution.
Depuis la racine du projet :
```bash
# Construire l'image (si Dockerfile est à la racine et concerne l'API)
docker build -t mon-api-segmentation -f Dockerfile . 
# Si le Dockerfile est dans api/ et est nommé Dockerfile:
# docker build -t mon-api-segmentation -f api/Dockerfile .

# Lancer le conteneur
docker run -p 8000:8000 mon-api-segmentation
```
L'API sera accessible sur `http://localhost:8000`.

### 5. Lancer l'Application Streamlit Localement

Assurez-vous que l'API FastAPI (locale ou déployée) est en cours d'exécution et que la variable `API_URL` dans `webapp/app_streamlit.py` pointe vers la bonne adresse.
```bash
cd webapp
streamlit run app_streamlit.py
```
L'application Streamlit s'ouvrira dans votre navigateur (généralement `http://localhost:8501`).

### 6. Exécuter les Tests Unitaires de l'API

Depuis la racine du projet (avec l'environnement de l'API activé) :
```bash
pytest api/tests/ -v
```

## ☁️ Déploiement sur Azure

L'API FastAPI est conçue pour être déployée sur Azure App Service via une image Docker stockée dans Azure Container Registry. Le déploiement est automatisé via un workflow GitHub Actions situé dans `.github/workflows/deploy_api_to_azure.yml`.

*   **URL de l'API Déployée :** `http://MonAPIsegmentationUnique.azurewebsites.net` 
*   **URL de l'Application Streamlit Déployée :** `http://MonAppStreamlitUnique.azurewebsites.net`
  

## 📊 Entraînement du Modèle

Les détails de l'exploration des données, de la préparation des générateurs de données, de la définition des modèles (U-Net Mini, ResNet50-UNet), de l'entraînement et de l'évaluation des performances (Dice, mIoU) se trouvent dans le notebook Jupyter :
*   `notebooks/notebook_modélisation_p8.ipynb`

Le suivi des expériences d'entraînement est géré avec MLflow.

## 🚧 Pistes d'Amélioration et Travaux Futurs

*   Amélioration continue des performances du modèle de segmentation (plus de données, augmentation plus poussée, architectures différentes, fine-tuning).
*   Optimisation de l'image Docker pour réduire sa taille.
*   Déploiement de l'application Streamlit sur Azure de manière plus formelle (par exemple, avec son propre App Service et CI/CD).
*   Ajout de tests d'intégration et de tests de performance pour l'API.
*   Mise en place d'un monitoring plus avancé sur Azure (Application Insights).

## Auteur

*   **Mathis Carré** - ([M-Carre sur GitHub](https://github.com/M-Carre))
