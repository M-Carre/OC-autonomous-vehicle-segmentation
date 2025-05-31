import streamlit as st
from PIL import Image
import requests
import io
import os # Ajouté pour lister les fichiers

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/predict/"
SAMPLE_IMAGE_DIR = "sample_test_images" # Chemin vers votre dossier d'images d'exemple

# --- Fonctions Utilitaires ---
def get_sample_image_names():
    """Retourne une liste des noms de fichiers dans le dossier SAMPLE_IMAGE_DIR."""
    if not os.path.exists(SAMPLE_IMAGE_DIR):
        return []
    return [f for f in os.listdir(SAMPLE_IMAGE_DIR) if os.path.isfile(os.path.join(SAMPLE_IMAGE_DIR, f))]

def process_and_display(image_bytes, image_name, image_type="image/png"):
    """Factorise la logique d'envoi à l'API et d'affichage des résultats."""
    st.image(image_bytes, caption=f"Image Sélectionnée : {image_name}", use_column_width=True)
    st.markdown("---")
    st.subheader("Résultat de la Segmentation :")

    with st.spinner("🧠 Analyse de l'image en cours... Veuillez patienter."):
        try:
            files = {"image_file": (image_name, image_bytes, image_type)}
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                predicted_mask_bytes = response.content
                st.image(predicted_mask_bytes, caption="Masque de Segmentation Prédit", use_column_width=True)
            else:
                try:
                    error_details = response.json()
                    st.error(f"Erreur de l'API ({response.status_code}): {error_details.get('detail', 'Erreur inconnue')}")
                except ValueError:
                    st.error(f"Erreur de l'API ({response.status_code}): {response.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"⚠️ Erreur de Connexion : Impossible de joindre l'API FastAPI à {API_URL}")
        except Exception as e:
            st.error(f"Une erreur inattendue est survenue : {e}")

# --- Interface Utilisateur Streamlit ---
st.title("👁️‍🗨️ Démonstration de Segmentation d'Images Cityscapes 🚗")
st.markdown("""
Bienvenue ! Choisissez une image d'exemple ou téléchargez la vôtre pour tester la segmentation.
""")

# Options pour la source de l'image
image_source_options = ["Télécharger une image"]
sample_images = get_sample_image_names()
if sample_images:
    image_source_options.extend(sample_images) # Ajoute les noms des images d'exemple

# st.sidebar.header("Options de Test") # Optionnel : mettre dans la barre latérale
selected_option = st.selectbox("Choisissez une source d'image :", image_source_options)

uploaded_file = None # Initialiser

if selected_option == "Télécharger une image":
    uploaded_file = st.file_uploader("Choisissez une image...", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if uploaded_file is not None:
        image_bytes_to_process = uploaded_file.getvalue()
        process_and_display(image_bytes_to_process, uploaded_file.name, uploaded_file.type)
else: # Une image d'exemple a été sélectionnée
    if selected_option in sample_images: # Vérifier que l'option est bien une image d'exemple
        image_path = os.path.join(SAMPLE_IMAGE_DIR, selected_option)
        try:
            with open(image_path, "rb") as f:
                image_bytes_to_process = f.read()
            # Déterminer le type MIME basiquement à partir de l'extension
            file_extension = os.path.splitext(selected_option)[1].lower()
            mime_type = "image/png" if file_extension == ".png" else "image/jpeg"

            process_and_display(image_bytes_to_process, selected_option, mime_type)
        except FileNotFoundError:
            st.error(f"L'image d'exemple '{selected_option}' n'a pas été trouvée.")
        except Exception as e:
            st.error(f"Erreur lors du chargement de l'image d'exemple : {e}")

if selected_option == "Télécharger une image" and uploaded_file is None:
    st.info("☝️ Veuillez télécharger une image ou sélectionner un exemple dans la liste.")


st.markdown("---")
st.markdown("Projet 8 - OpenClassrooms - Traitement d'images pour véhicule autonome")