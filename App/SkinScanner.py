import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import tensorflow_hub as hub

# Carica il modello
reloaded = tf.keras.models.load_model("./Modello", custom_objects={'KerasLayer': hub.KerasLayer})
labels = ['benign', 'malignant']

# Variabili globali per l'immagine e il risultato
image_path = ""
predicted_class = ""
confidence = 0.0

# Funzione per effettuare la previsione sull'immagine
def predict_image(image):
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)
    prediction = reloaded.predict(input_image)
    class_index = np.argmax(prediction)
    predicted_class = labels[class_index]
    confidence = prediction[0][class_index]
    return predicted_class, confidence

# Funzione chiamata quando viene premuto il pulsante "Carica immagine"
def open_file_dialog():
    global image_path, predicted_class, confidence
    file_path = filedialog.askopenfilename()
    if file_path:
        image_path = file_path
        predicted_class = ""
        confidence = 0.0
        show_image()

# Funzione chiamata quando viene premuto il pulsante "Classifica"
def classify_image():
    global image_path, predicted_class, confidence
    if image_path:
        image = cv2.imread(image_path)
        predicted_class, confidence = predict_image(image)
        show_result()

# Funzione per mostrare l'immagine
def show_image():
    image = Image.open(image_path)
    image = image.resize((400, 400))
    photo = ImageTk.PhotoImage(image)
    image_label.configure(image=photo)
    image_label.image = photo

# Funzione per mostrare il risultato
def show_result():
    result_label.configure(text=f"Classe: {predicted_class}\nConfidenza: {confidence:.2f}")

# Funzione per mostrare i credits
def show_credits():
    tk.messagebox.showinfo("Credits", "App di Classificazione delle Immagini")

# Crea la finestra principale dell'applicazione
root = tk.Tk()
root.title("SkinScanner")
root.geometry("500x700")

# Configura lo stile
root.configure(bg="#EAF6FF")

# Cambia lo stile dei pulsanti
button_style = {
    "font": ("Arial", 12),
    "foreground": "white",
    "background": "#2E5894",
    "relief": "flat",
    "borderwidth": 0,
    "padx": 10,
    "pady": 5
}

# Frame per l'immagine
image_frame = tk.Frame(root, bg="#EAF6FF")
image_frame.pack(pady=10)

# Etichetta per il titolo
title_label = tk.Label(image_frame, text="SkinScanner", font=("Arial", 16), bg="#EAF6FF")
title_label.pack()

# Pulsante per caricare un'immagine
open_button = tk.Button(image_frame, text="Carica immagine", command=open_file_dialog, **button_style)
open_button.pack(pady=10)

# Spazio per visualizzare l'immagine
image_label = tk.Label(image_frame, bg="white", relief="solid", borderwidth=1)
image_label.pack(pady=10)

# Mostra la foto di default
default_image = Image.open("default.png")
default_image = default_image.resize((400, 409))
default_photo = ImageTk.PhotoImage(default_image)
image_label.configure(image=default_photo)
image_label.image = default_photo

# Frame per il risultato
result_frame = tk.Frame(root, bg="#EAF6FF")
result_frame.pack(pady=20)

# Pulsante per classificare l'immagine
classify_button = tk.Button(result_frame, text="Classifica", command=classify_image, **button_style)
classify_button.pack(pady=10)

# Etichetta per il risultato
result_label = tk.Label(result_frame, font=("Arial", 12), bg="#EAF6FF")
result_label.pack()

# Avvia l'applicazione
root.mainloop()
