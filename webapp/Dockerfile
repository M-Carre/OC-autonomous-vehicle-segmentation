# webapp/Dockerfile

# 1. Utiliser une image Python de base
FROM python:3.10-slim

# 2. Définir le répertoire de travail dans le conteneur
WORKDIR /app

# 3. Copier le fichier des dépendances
#    On suppose que requirements_streamlit.txt sera à la racine de webapp/
COPY requirements_streamlit.txt ./requirements_streamlit.txt

# 4. Installer les dépendances
RUN pip install --no-cache-dir -r requirements_streamlit.txt

# 5. Copier tous les fichiers de l'application Streamlit
#    (app_streamlit.py et le dossier sample_test_images/)
COPY . .
# Cette commande copie tout ce qui se trouve dans le dossier webapp/ (le contexte de build
# pour ce Dockerfile si on le build depuis le dossier webapp) vers /app dans l'image.
# Si vous buildez depuis la racine du projet, il faudra ajuster les chemins de copie.

# 6. Exposer le port par défaut de Streamlit (8501)
EXPOSE 8501

# 7. Commande pour lancer l'application Streamlit
#    Streamlit s'attend à ce que le serveur écoute sur toutes les interfaces (0.0.0.0)
#    et le port 8501 sera mappé par App Service.
#    --server.headless true est recommandé pour les environnements conteneurisés
#    pour éviter que Streamlit n'essaie d'ouvrir un navigateur.
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]