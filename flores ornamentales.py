import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'C:/Users/user/Documents/ornamentales', # Usa tu propia ruta -por favor-
        target_size=(320, 240),
        batch_size=20,
        class_mode='categorical') 

test_generator = test_datagen.flow_from_directory(
        'C:/Users/user/Documents/ornamentales prueba', 
        target_size=(320, 240),
        batch_size=5,
        class_mode='categorical')

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(320, 240, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(5, activation='softmax') 
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',  
              metrics=['accuracy'])

model.fit(train_generator, epochs=100)

test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

window = tk.Tk()
window.title("Clasificador de flores ornamentales")

def load_and_predict():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = image.resize((320, 240))  
        image = img_to_array(image) / 255.0 
        image = np.expand_dims(image, axis=0) 
        prediction = model.predict(image)
        species = ['clavel', 'helecho', 'jacaranda', 'lirio', 'rosal'] # Si usas tu propio banco de imagenes cambia los nombres
        predicted_species = species[np.argmax(prediction)]
        label.config(text="Especie predicha: " + predicted_species)

# Interfaz
button = tk.Button(window, text="Cargar Imagen", command=load_and_predict)
button.pack()

label = tk.Label(window, text="")
label.pack()

window.mainloop()





