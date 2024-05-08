import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

# Paso 1: Preprocesamiento de datos
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'C:/Users/Alejandra Juárez/Documents/4to Semestre/Aplicaciones multimedia/ornamentales',
        target_size=(320, 240),
        batch_size=20,
        class_mode='categorical')  # Cambiado a 'categorical' para múltiples clases

test_generator = test_datagen.flow_from_directory(
        'C:/Users/Alejandra Juárez/Documents/4to Semestre/Aplicaciones multimedia/ornamentales prueba',
        target_size=(320, 240),
        batch_size=5,
        class_mode='categorical')  # Cambiado a 'categorical' para múltiples clases

# Paso 2: Definición del modelo
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
    layers.Dense(5, activation='softmax')  # Cambiado a 5 clases y 'softmax'
])

# Paso 3: Entrenamiento del modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Cambiado para múltiples clases
              metrics=['accuracy'])

model.fit(train_generator, epochs=100)

# Paso 4: Evaluación del modelo
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

# Crear una ventana para la interfaz gráfica
window = tk.Tk()
window.title("Clasificador de flores ornamentales")

# Función para cargar una imagen y realizar la predicción
def load_and_predict():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = image.resize((320, 240))  # Redimensionar la imagen para que coincida con el tamaño de entrada del modelo
        image = img_to_array(image) / 255.0  # Normalizar la imagen
        image = np.expand_dims(image, axis=0)  # Añadir dimensión del lote
        prediction = model.predict(image)
        species = ['clavel', 'helecho', 'jacaranda', 'lirio', 'rosal']
        predicted_species = species[np.argmax(prediction)]
        label.config(text="Especie predicha: " + predicted_species)

# Crear un botón para cargar una imagen
button = tk.Button(window, text="Cargar Imagen", command=load_and_predict)
button.pack()

# Etiqueta para mostrar la especie predicha
label = tk.Label(window, text="")
label.pack()

# Ejecutar la aplicación
window.mainloop()





