# api/Dockerfile

# 1. Image de base Python
FROM python:3.10-slim

# 2. Répertoire de travail
WORKDIR /app

# 3. Copier et installer les dépendances de l'API
COPY api/requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# 4. Copier le code source de l'API
# Copie le contenu du dossier 'api' (du contexte de build) vers '/app' dans l'image
COPY api/main.py /app/main.py 

# 5. Copier les modèles
# Copie le contenu du dossier 'models' (de la racine du contexte de build)
# vers '/app/models/' dans l'image.
# Cela suppose que api/main.py aura MODEL_PATH = "models/NOM_DU_MODELE.keras"
COPY models/ /app/models/

# 6. Exposer le port (informatif, Azure App Service utilisera $PORT)
EXPOSE 8000

# 7. Commande de démarrage pour Gunicorn avec Uvicorn workers
# Gunicorn se liera au port $PORT fourni par Azure App Service lors du déploiement.
# Pour le test local, la variable $PORT n'existera pas, donc Gunicorn utilisera 8000 par défaut ici.
# Si vous voulez être explicite pour le local : ["gunicorn", ..., "--bind", "0.0.0.0:8000"]
# Si vous voulez préparer pour Azure qui injecte $PORT:
# Vous pouvez utiliser un script de démarrage (entrypoint.sh) qui gère cela,
# ou Azure App Service vous permet de spécifier la commande de démarrage dans sa configuration.
# La commande CMD ici est souvent celle pour le test local ou une commande de base.
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000", "--timeout", "120"]
