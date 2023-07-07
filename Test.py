#file per testare il modello e caricare una immagine
#caricare il modello dalla sua directory
#modificare il nome dell' immagine da inserire
import cv2
import numpy as np
import tensorflow as tf

reloaded = tf.keras.models.load_model("./Modello", custom_objects={'KerasLayer':hub.KerasLayer})
def predict_reload(image):
    probabilities = reloaded.predict(np.asarray([img]))[0]
    class_idx = np.argmax(probabilities)
    return {Labels[class_idx]: probabilities[class_idx]}


Labels=["benign", "malignant"]
# Carica l'immagine
image = cv2.imread("image.jpg") #image da sostituire con l'immagine che si vuole classificare

# Ridimensiona l'immagine alle dimensioni desiderate
resized_image = cv2.resize(image, (224, 224))

# Normalizza i valori dei pixel nell'intervallo [0, 1]
normalized_image = resized_image / 255.0

# Aggiungi una dimensione batch all'immagine
input_image = np.expand_dims(normalized_image, axis=0)

# Esegue la previsione sul modello
prediction = reloaded.predict(input_image)

# Ottieni l'etichetta della classe predetta
class_index = np.argmax(prediction)
predicted_class = Labels[class_index]

# Ottieni la confidenza della classe predetta
confidence = prediction[0][class_index]

# Stampa il risultato della previsione
print("Classe predetta:", predicted_class)
print("Confidenza:", confidence)
