# main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse # Pour retourner une image
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image # Python Imaging Library
import io # Pour gérer les flux de bytes en mémoire
import traceback
from tensorflow.keras.utils import register_keras_serializable
import os

# --- Configuration et Chargement du Modèle (au démarrage de l'application) ---

# Obtenir le chemin du dossier où se trouve main.py (c'est-à-dire api/)
API_DIR = os.path.dirname(os.path.abspath(__file__))

# Construire le chemin vers le dossier racine du projet (un niveau au-dessus de api/)
PROJECT_ROOT = os.path.dirname(API_DIR) 

# Construire le chemin vers le modèle
MODEL_NAME = "UNetMini_DevSet_NoAug_AggressiveCB_best.keras"
MODEL_PATH = f"models/{MODEL_NAME}" 

print(f"Chemin absolu calculé pour le modèle : {MODEL_PATH}") # Pour débogage

TARGET_IMG_WIDTH_API = 512
TARGET_IMG_HEIGHT_API = 256

# Supposons que NUM_TARGET_CLASSES est défini globalement (ex: 8)
NUM_TARGET_CLASSES = 8 # Doit correspondre à votre modèle

model = None # Variable globale pour le modèle

app = FastAPI(title="API de Segmentation d'Images Cityscapes")

# Définition de la palette de couleurs au niveau global
CITYSCAPES_COLOR_MAP_8_CLASSES = {
    0: (128, 64, 128),  # 0: flat (road, sidewalk) - Mauve
    1: (220, 20, 60),   # 1: human (person, rider) - Rouge
    2: (0, 0, 142),     # 2: vehicle (car, truck, bus, ...) - Bleu foncé
    3: (70, 70, 70),    # 3: construction (building, wall, fence) - Gris foncé
    4: (220, 220, 0),   # 4: object (pole, traffic light/sign) - Jaune
    5: (107, 142, 35),  # 5: nature (vegetation, terrain) - Vert olive
    6: (70, 130, 180),  # 6: sky - Bleu ciel
    7: (0, 0, 0)        # 7: void/ignore - Noir
}

# --- Métrique : Dice Coefficient (Simplifiée) ---
@register_keras_serializable(package='Custom', name='DiceCoefficientSimplified') # DÉCORATEUR AJOUTÉ
def dice_coefficient_metric_simplified(y_true, y_pred, smooth=1e-6):
    """
    Calcule le Dice Coefficient. y_true contient des ID de classe 0 à NUM_TARGET_CLASSES-1.
    y_pred contient des probabilités (après softmax).
    """
    y_true_int = tf.cast(y_true, tf.int32)
    # Convertir y_true en one-hot car Dice compare les probabilités/masques binaires par classe
    y_true_one_hot = tf.one_hot(y_true_int, depth=NUM_TARGET_CLASSES, axis=-1, dtype=tf.float32)
    y_pred_probs = tf.nn.softmax(y_pred, axis=-1) # Assurer que y_pred sont des probabilités

    # Aplatir les dimensions spatiales
    y_true_flat = tf.reshape(y_true_one_hot, [-1, NUM_TARGET_CLASSES])
    y_pred_flat = tf.reshape(y_pred_probs, [-1, NUM_TARGET_CLASSES])

    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0) # Par classe
    sum_y_true = tf.reduce_sum(y_true_flat, axis=0)
    sum_y_pred = tf.reduce_sum(y_pred_flat, axis=0)

    dice_per_class = (2. * intersection + smooth) / (sum_y_true + sum_y_pred + smooth)
    mean_dice = tf.reduce_mean(dice_per_class) # Moyenne sur les classes
    return mean_dice

# --- Métrique : Mean Intersection over Union (mIoU) Personnalisée ---
# NUM_TARGET_CLASSES doit être défini globalement (valeur 8)
@register_keras_serializable(package='Custom', name='MeanIoUCustom') # DÉCORATEUR AJOUTÉ
def mean_iou_custom(y_true, y_pred, smooth=1e-6):
    """
    Calcule le Mean Intersection over Union (mIoU) comme métrique.
    y_true contient des ID de classe 0 à NUM_TARGET_CLASSES-1.
    y_pred contient des probabilités (après softmax) ou des logits.
    """
    # 1. Prétraiter y_true:
    #    - S'assurer qu'il est de type entier.
    #    - Convertir en one-hot.
    y_true_int = tf.cast(y_true, tf.int32)
    y_true_one_hot = tf.one_hot(y_true_int, depth=NUM_TARGET_CLASSES, axis=-1, dtype=tf.float32)
    # y_true_one_hot shape: (batch_size, height, width, NUM_TARGET_CLASSES)

    # 2. Prétraiter y_pred:
    #    - Appliquer softmax si ce ne sont pas déjà des probabilités.
    #    - Pour IoU, on compare souvent les masques binaires (prédit vs réel) par classe.
    #      Donc, on prend l'argmax des probabilités pour obtenir les ID de classe prédits,
    #      puis on les reconvertit en one-hot pour la comparaison par classe.
    y_pred_probs = tf.nn.softmax(y_pred, axis=-1)
    y_pred_labels = tf.argmax(y_pred_probs, axis=-1, output_type=tf.int32) # Shape: (batch, H, W)
    y_pred_one_hot = tf.one_hot(y_pred_labels, depth=NUM_TARGET_CLASSES, axis=-1, dtype=tf.float32)
    # y_pred_one_hot shape: (batch_size, height, width, NUM_TARGET_CLASSES)

    # 3. Aplatir les dimensions spatiales
    y_true_flat = tf.reshape(y_true_one_hot, [-1, NUM_TARGET_CLASSES])
    y_pred_flat = tf.reshape(y_pred_one_hot, [-1, NUM_TARGET_CLASSES])
    # Nouvelles formes : (batch_size * height * width, NUM_TARGET_CLASSES)

    # 4. Calculer l'Intersection et l'Union par classe
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0) # TP par classe, Shape: (NUM_TARGET_CLASSES,)

    sum_y_true = tf.reduce_sum(y_true_flat, axis=0) # TP + FN par classe
    sum_y_pred = tf.reduce_sum(y_pred_flat, axis=0) # TP + FP par classe

    union = sum_y_true + sum_y_pred - intersection # Union = (TP+FN) + (TP+FP) - TP = TP+FN+FP

    iou_per_class = (intersection + smooth) / (union + smooth) # IoU par classe, Shape: (NUM_TARGET_CLASSES,)

    # Mean IoU sur les classes.
    # Pour la simplicité et la robustesse initiale, moyennons sur toutes les classes cibles :
    mean_iou = tf.reduce_mean(iou_per_class)

    return mean_iou


# --- Fonction de Perte : Dice Loss (Simplifiée) ---
def dice_loss_simplified(y_true, y_pred, smooth=1e-6):
    """
    Calcule la Dice Loss. y_true contient des ID de classe 0 à NUM_TARGET_CLASSES-1.
    y_pred contient des probabilités (après softmax).
    Loss = 1 - Mean Dice Coefficient.
    """
    # Réutilise la logique de la métrique Dice (sans la dernière moyenne si on veut la perte par classe)
    # ou directement 1 - la métrique.
    # Pour la perte, il est courant de vouloir une valeur par échantillon du batch,
    # puis Keras fait la moyenne sur le batch.
    # Ici, nous calculons le Mean Dice sur le batch et retournons 1 - Mean Dice.

    # Pour être plus précis pour une fonction de perte, on calcule souvent le Dice par image du batch,
    # puis on moyenne ces pertes. Mais pour commencer, 1 - mean_dice_coeff_on_batch est acceptable.

    mean_dice = dice_coefficient_metric_simplified(y_true, y_pred, smooth)
    loss = 1.0 - mean_dice
    return loss


# --- Fonction de Perte : Sparse Categorical Cross-Entropy (de Keras) ---
# Elle fonctionnera maintenant directement car y_true ne contient que des ID valides [0, NUM_CLASSES-1]
sparse_ce_loss_fn_simplified = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


# --- Fonction de Perte : Combinée (SparseCE + Dice) - Simplifiée ---
@register_keras_serializable(package='Custom', name='CombinedDiceSparseCELossSimplified')
class CombinedDiceSparseCELossSimplified(tf.keras.losses.Loss): 
    def __init__(self, alpha=0.5, beta=0.5, smooth_dice=1e-6, 
                 name="combined_dice_sparse_ce_loss_simplified", 
                 reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, # AJOUTÉ
                 **kwargs):                                # AJOUTÉ
        super().__init__(name=name, reduction=reduction, **kwargs) # MODIFIÉ
        self.alpha = alpha
        self.beta = beta
        self.smooth_dice = smooth_dice
        self.sparse_ce_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False 
            # La réduction sera gérée par la classe Loss parente si cette instance
            # est configurée avec reduction=NONE, sinon elle fait sa propre réduction.
            # Laissons la réduction par défaut ici pour sparse_ce_loss_fn, qui est SUM_OVER_BATCH_SIZE.
        )

    def call(self, y_true, y_pred):
        y_true_int = tf.cast(y_true, tf.int32)
        s_ce_loss = self.sparse_ce_loss_fn(y_true_int, y_pred) # Retourne un scalaire (perte moyenne du batch)
        d_loss = dice_loss_simplified(y_true, y_pred, smooth=self.smooth_dice) # Retourne un scalaire
        total_loss = (self.alpha * s_ce_loss) + (self.beta * d_loss)
        return total_loss # Retourne un scalaire

    def get_config(self): # AJOUTÉ
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "beta": self.beta,
            "smooth_dice": self.smooth_dice
        })
        return config


CUSTOM_OBJECTS = {
    'CombinedDiceSparseCELossSimplified': CombinedDiceSparseCELossSimplified,
    'dice_coefficient_metric_simplified': dice_coefficient_metric_simplified,
    'mean_iou_custom': mean_iou_custom
}


@app.on_event("startup") # Événement exécuté au démarrage de FastAPI

async def load_application_model():
    global model
    try:
        print(f"Chargement du modèle depuis : {MODEL_PATH}") # Affiche le chemin utilisé
        # AJOUTEZ CECI POUR DÉBOGAGE :
        print(f"Répertoire de travail actuel (au chargement) : {os.getcwd()}")
        print(f"Chemin absolu tenté pour le modèle : {os.path.abspath(MODEL_PATH)}")
        if not os.path.exists(MODEL_PATH):
            print(f"ERREUR : Le fichier modèle N'EXISTE PAS à {os.path.abspath(MODEL_PATH)}")
            # Lister le contenu de /app et /app/models pour voir ce qui est là
            print("Contenu de /app :")
            os.system("ls -l /app")
            print("Contenu de /app/models :")
            os.system("ls -l /app/models")

        model = tf.keras.models.load_model(MODEL_PATH, 
                                           custom_objects=CUSTOM_OBJECTS, 
                                           compile=False) # compile=False car nous n'avons pas besoin de compiler le modèle ici,
        model.summary() # Pour vérifier qu'il est chargé
        print("Modèle chargé avec succès !")
    except Exception as e:
        print(f"ERREUR lors du chargement du modèle : {e}") # CE MESSAGE DEVRAIT APPARAÎTRE DANS VOS LOGS DOCKER
        print(traceback.format_exc()) # Affiche la trace complète de l'erreur
        model = None

    

@app.get("/")
async def read_root():
    return {"message": "Bienvenue à l'API de segmentation d'images ! Utilisez le endpoint /predict/ pour la segmentation."}

# --- Endpoint de Prédiction (Squelette) ---
@app.post("/predict/",
          summary="Prédit le masque de segmentation pour une image.",
          response_description="L'image du masque de segmentation prédit (PNG).")
async def predict_segmentation(image_file: UploadFile = File(..., description="Fichier image à segmenter.")):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé ou erreur lors du chargement.")

    if not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Type de fichier invalide. Seules les images sont acceptées.")

    try:
        contents = await image_file.read()
        pil_image = Image.open(io.BytesIO(contents))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        img_np_original = np.array(pil_image)

        target_dsize_api = (TARGET_IMG_WIDTH_API, TARGET_IMG_HEIGHT_API)
        img_resized = cv2.resize(img_np_original, target_dsize_api, interpolation=cv2.INTER_LINEAR)
        
        # Forcer float32 pour la normalisation pour être cohérent
        img_normalized = img_resized.astype(np.float32) / 255.0 
        
        img_batch = np.expand_dims(img_normalized, axis=0)
        print(f"Image prétraitée, shape pour le modèle : {img_batch.shape}, dtype: {img_batch.dtype}")

        # --- 1. Inférence avec le modèle ---
        predictions_raw = model.predict(img_batch)
        print(f"Prédictions brutes obtenues, shape: {predictions_raw.shape}, dtype: {predictions_raw.dtype}")

        # --- 2. Post-traitement (argmax) ---
        predicted_mask_labels = np.argmax(predictions_raw[0], axis=-1).astype(np.uint8) # Assurer uint8
        print(f"Masque de labels prédits généré, shape: {predicted_mask_labels.shape}, dtype: {predicted_mask_labels.dtype}")
        print(f"Valeurs uniques dans le masque de labels: {np.unique(predicted_mask_labels)}")

        # --- 3. Convertir le masque de labels en image couleur PNG ---
        height, width = predicted_mask_labels.shape
        output_rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

        for class_id, color in CITYSCAPES_COLOR_MAP_8_CLASSES.items(): # Utilise la map globale
            output_rgb_mask[predicted_mask_labels == class_id] = color

        pil_output_mask_image = Image.fromarray(output_rgb_mask, mode='RGB')
        img_byte_arr = io.BytesIO()
        pil_output_mask_image.save(img_byte_arr, format='PNG')
        img_byte_arr_value = img_byte_arr.getvalue() # Obtenir les bytes

        return StreamingResponse(io.BytesIO(img_byte_arr_value), media_type="image/png")

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Erreur détaillée lors du traitement de l'image : {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur lors du traitement de l'image : {str(e)}")