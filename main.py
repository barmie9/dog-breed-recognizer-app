import tkinter as tk

from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

class DogBreedDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozpoznawanie rasy psa")


        self.root.minsize(width=850, height=650)  # Ustawia minimalny rozmiar okna na 850x650 pikseli
        self.root.configure(bg='#616161')  # Ustawia kolor tła okna

        # Wczytywanie domyslnego zdjecia
        self.image = Image.open('dog_anim.jpg')
        photo = ImageTk.PhotoImage(self.image)


        # Ładowanie modelu
        self.load_model()

        #Utworzenie "labeli" interfejsu graficznego:

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)
        # Przypisanie domyslnego zdjecia do image_label
        self.image_label.configure(image=photo)
        self.image_label.image = photo

        self.load_button = tk.Button(root, text="Wczytaj Zdjęcie", command=self.load_image, bg='blue', fg='white', font=('Arial', 14))
        self.load_button.pack(pady=10)

        self.detect_button = tk.Button(root, text="Wykryj Rasę", command=self.detect_breed, bg='blue', fg='white', font=('Arial', 14))
        self.detect_button.pack(pady=10)

        self.result_label = tk.Label(root, text="- ? -", bg='#616161', fg='#07f533', font=('Arial', 16, 'bold'))
        self.result_label.pack(pady=10)

        self.other_result_label_ = tk.Label(root, text="Innne możliwości:", bg='#616161', font=('Arial', 14))
        self.other_result_label_.pack(pady=10)

        self.other_result_label = tk.Label(root, text="", bg='#616161', fg='#fca335', font=('Arial', 13, 'bold'))
        self.other_result_label.pack(pady=10)

    # Funkcja ładująca wyuczony model z pliku
    def load_model(self):
        model_path = "inference_graph_v3/saved_model"
        self.model = tf.saved_model.load(model_path)

    # Funkcja wczytujęca zdjęcie z pliku
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image = Image.open(file_path)
            self.display_image()
            (im_width, im_height) = self.image.size
            self.image_numpy_array = np.array(self.image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    # Funkcja wyświetlająca wczytane zdjęcie
    def display_image(self):
        image = self.image.resize((300, 300), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    # Najważniejsza funkcja rozpoznająca rase psa na podstawie wczytanego modelu oraz zdjecia
    def detect_breed(self):
        if hasattr(self, 'image'):
            input_tensor = self.preprocess_image()
            # Rozpoznawanie obiektów:
            predictions = self.model(input_tensor)

            # Pobranie z 'predictions' informacji na temat rozpoznanych obiektów
            breed_id = int(predictions['detection_classes'][0][0].numpy())
            breed_name = self.get_breed_name(breed_id)
            confidence = predictions['detection_scores'][0][0] * 100

            # Jeżeli prawdopodobieństwo rozpoznanej rasy jest większe od 50% to jest ona wyświetlana
            if confidence > 50:
                result_text = f"Rasa: {breed_name}\nPrawdopodobieństwo: {confidence:.2f}%"
            else:
                result_text = "Nie rozpoznano rasy"
            self.result_label.config(text=result_text)

            # Wyświetlenie innych rozpoznanych ras :
            other_breed = ""
            for i in range(3):
                conf = predictions['detection_scores'][0][i] * 100
                other_breed += self.get_breed_name(int(predictions['detection_classes'][0][i].numpy())) + f" ({conf:.2f}%) , "
            self.other_result_label.config(text=other_breed)



    # Funkcja przetwarzająca obraz do detekcji:
    def preprocess_image(self):
        input_tensor = tf.convert_to_tensor(self.image_numpy_array)

        # Model oczekuje 'batch' obrazów, dlatego dodajemy oś za pomocą tf.newaxis.
        input_tensor = input_tensor[tf.newaxis, ...]
        return input_tensor

    # Funkcja konwertująca id rasy na jego nazwe:
    def get_breed_name(self, breed_id):
        label_map = {
            1: 'Doberman',
            2: 'Owczarek niemiecki',
            3: 'Golden retriever',
            4: 'Sznaucer',
            5: 'Yorkshire terrier',
            6: 'Berneński pies pasterski',
            7: 'Chihuahua',
            8: 'Buldog francuski',
            9: 'Poodle',
            10: 'Syberyjski Husky',
            11: 'Beagle',
            12: 'Border collie',
            13: 'Komondor',
            14: 'Rottweiler',
            15: 'Shih tzu'
        }
        return label_map.get(breed_id, 'Unknown')

# Główna funkcja programu:
if __name__ == "__main__":
    root = tk.Tk()
    app = DogBreedDetectorApp(root)
    root.mainloop()

