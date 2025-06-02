# OC-autonomous-vehicle-segmentation/api/tests/test_main_api.py

import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import numpy as np
import os
import sys

# Ajout de la racine du projet au path pour les tests
PROJECT_ROOT_FOR_TESTS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT_FOR_TESTS)

# Importer 'app' après avoir potentiellement moqué les dépendances si nécessaire,
# ou moquer les attributs de 'app' après son import.
from api.main import app, MODEL_PATH, CUSTOM_OBJECTS, TARGET_IMG_HEIGHT_API, TARGET_IMG_WIDTH_API, NUM_TARGET_CLASSES

# Classe factice pour simuler un modèle Keras
class MockKerasModel:
    def predict(self, batch_input):
        batch_size = batch_input.shape[0]
        h = TARGET_IMG_HEIGHT_API
        w = TARGET_IMG_WIDTH_API
        num_classes = NUM_TARGET_CLASSES
        mock_prediction = np.zeros((batch_size, h, w, num_classes), dtype=np.float32)
        mock_prediction[:, :, :, 0] = 1.0 # Classe 0 toujours prédite
        return mock_prediction

    def summary(self):
        print("MockKerasModel summary: This is a mock model.")

mock_model_instance = MockKerasModel()

# --- Fixture pour une image d'exemple ---
@pytest.fixture
def sample_image_bytes():
    """Crée une image factice en bytes pour les tests d'upload."""
    img = Image.new('RGB', (TARGET_IMG_WIDTH_API, TARGET_IMG_HEIGHT_API), color = 'black')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

# --- Fixture pour moquer le modèle APRÈS l'initialisation de l'app ---
# Cette fixture sera utilisée explicitement par les tests qui en ont besoin.
@pytest.fixture
def mock_api_model(monkeypatch):
    """Moque la variable globale 'model' dans le module api.main."""
    # Moquer la variable 'model' dans le module 'api.main'
    # Ceci est fait APRÈS que TestClient(app) ait potentiellement chargé le vrai modèle
    # via l'événement de démarrage. Nous le remplaçons pour les tests suivants.
    # Cela suppose que vous utilisez @app.on_event("startup")
    # Si vous utilisez lifespan avec app.state.model, la moquerie serait différente.
    
    # Pour importer le module et non juste l'objet 'app'
    import api.main as main_module 
    
    original_model = main_module.model # Sauvegarder le vrai modèle (ou None)
    monkeypatch.setattr(main_module, "model", mock_model_instance)
    print("MOCK: api.main.model is now a MockKerasModel for this test.")
    yield # Le test s'exécute
    # Restaurer après le test
    monkeypatch.setattr(main_module, "model", original_model)
    print("UNMOCK: api.main.model restored.")


# --- Client de Test ---
# Le TestClient est créé une fois ici. Son initialisation va déclencher
# l'événement de démarrage de l'application FastAPI et donc le chargement du modèle.
# Pour les tests unitaires qui ne veulent PAS de ce vrai chargement,
# la fixture `mock_api_model` le remplacera.
client = TestClient(app)


# --- Tests ---
def test_read_root():
    """Teste l'endpoint racine."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue à l'API de segmentation d'images ! Utilisez le endpoint /predict/ pour la segmentation."}

def test_predict_segmentation_success(sample_image_bytes, mock_api_model): # Utilise le mock explicite
    """Teste l'endpoint de prédiction avec une image valide et un modèle moqué."""
    # mock_api_model est une fixture qui s'assure que api.main.model est notre mock_model_instance
    
    files = {"image_file": ("test_image.png", io.BytesIO(sample_image_bytes), "image/png")}
    response = client.post("/predict/", files=files)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    
    try:
        img_response = Image.open(io.BytesIO(response.content))
        assert img_response.format == "PNG"
        assert img_response.size == (TARGET_IMG_WIDTH_API, TARGET_IMG_HEIGHT_API)
        # Puisque notre mock prédit toujours la classe 0 (mauve [128, 64, 128])
        img_array = np.array(img_response)
        expected_color_class_0 = np.array([128, 64, 128]) 
        assert np.all(img_array[0,0] == expected_color_class_0)
    except Exception as e:
        pytest.fail(f"L'image PNG retournée n'a pas pu être validée : {e}")

def test_predict_segmentation_invalid_file_type(mock_api_model): # Utilise le mock, car on ne teste pas le modèle ici
    """Teste l'endpoint de prédiction avec un type de fichier invalide."""
    fake_text_file_content = b"Ceci n'est pas une image."
    files = {"image_file": ("test.txt", io.BytesIO(fake_text_file_content), "text/plain")}
    response = client.post("/predict/", files=files)
    
    assert response.status_code == 400
    assert response.json()["detail"] == "Type de fichier invalide. Seules les images sont acceptées."

def test_predict_segmentation_no_model_truly_loaded(monkeypatch, sample_image_bytes):
    """
    Teste le cas où la variable globale 'model' serait None.
    Cette approche moque directement la variable 'model' dans api.main.
    """
    import api.main as main_module
    
    original_model = main_module.model # Sauvegarder le modèle actuel (qui pourrait être le vrai ou un mock)
    monkeypatch.setattr(main_module, "model", None) # Forcer le modèle à être None
    print("MOCK: api.main.model is now None for this test.")

    files = {"image_file": ("test_image.png", io.BytesIO(sample_image_bytes), "image/png")}
    response = client.post("/predict/", files=files)
    
    assert response.status_code == 503
    assert response.json()["detail"] == "Modèle non chargé ou erreur lors du chargement."

    monkeypatch.setattr(main_module, "model", original_model) # Restaurer
    print("UNMOCK: api.main.model restored after no_model_truly_loaded test.")