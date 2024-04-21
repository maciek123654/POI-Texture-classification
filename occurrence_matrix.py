import os
import numpy as np
import pandas as pd
from skimage import io, color
from skimage.util import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
import tkinter as tk
from tkinter import filedialog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# wyświetla okno dialogowe pozwalające użytkownikowi wybrać folder,
# a następnie zwraca wybraną ścieżkę
def select_folder_dialog(prompt):
    root = tk.Tk()  # Tworzy instancję głównego okna
    root.withdraw()  # Ukrywa główne okno
    folder_selected = filedialog.askdirectory(title=prompt)
    root.destroy()
    return folder_selected  # Zwraca wybraną ścieżkę do folderu

# wczytuje obrazy z podanego folderu i dzieli je na mniejsze fragmenty o określonym rozmiarze
def load_and_slice_images(directory, size):
    sliced_images = []

    for filename in os.listdir(directory):  # Iteruje przez pliki w podanym katalogu
        img = io.imread(os.path.join(directory, filename))  # Wczytuje obraz
        for i in range(0, img.shape[0], size):  # Iteruje po wierszach obrazu
            for j in range(0, img.shape[1], size):  # Iteruje po kolumnach obrazu
                if i + size <= img.shape[0] and j + size <= img.shape[1]:  # Sprawdza, czy można wyodrębnić fragment
                    sliced_images.append(img[i:i + size, j:j + size])  # Dodaje wyodrębniony fragment do listy

    return sliced_images

# oblicza cechy tekstury dla podanych obrazów, korzystając z macierzy współwystępowania
# szarości (GLCM) i zwraca wektor cech
def calculate_texture_features(images, distances, angles, properties):
    features_list = []

    for img in images:
        gray = color.rgb2gray(img)  # konwersja na skale szarości
        gray = img_as_ubyte(gray)  # konwersja do 8 - bitowej głębi kolorów
        gray //= 4 # konwersja liczby poziomów szarości do 64
        # macierz współwystępowania poziomów szarości (GLCM)
        glcm = graycomatrix(gray, distances=distances, angles=angles, levels=64, symmetric=True, normed=True)
        feature_vector = []
        for prop in properties:  # Iteracja po liście nazw cech, które mają być wyliczone
            feature_vector.extend([graycoprops(glcm, prop).ravel()])  # ravel() jest używana do spłaszczenia tablicy do jednego wymiaru
        features_list.append(np.concatenate(feature_vector))
    return np.array(features_list)

selected_folders = [select_folder_dialog(f'Wybierz katalog tekstury') for i in range(3)]

all_features = []
labels = []
for folder in selected_folders:
    images = load_and_slice_images(folder, 128)  # Wczytuje i dzieli obrazy na fragmenty
    # Definicja odległości i kierunków
    # małe odległości - 1px, znajdowanie subtelnych różnic
    # większe odległości - 5px,  analiza dużych obszarów
    # 0 stopni - 0
    # 45 stopni - np.pi/4
    # 90 stopni - np.pi/2
    # 135 stopni - 3*np.pl/4, uzględnianie poziomych, pionowych i ukośnych tekstur
    # - dissimilarity odnosi się do odmienności tekstury na obrazie
    # mierzy to średnią różnicę w intensywności między pikselami w obrazie
    # im wyższa wartość tym bardziej zróżnicowana jest tekstura
    # - correlation odnosi się do korelacji między intensywnościami pikseli w obrazie
    # im wyższa wartość tym bardziej liniowo zależne są piksele w teksturze
    # - contrast odnosi się do kontrastu między intensywnoścami pikseli w analizowanej teksturze
    # im wyższa wartość tym większa różnica w intensywności pikseli i większy kontrast tekstury na obrazie
    # - energy mierzy sumę kwadratów wartości pikseli w macierzy
    # jest to miara jednorodności lub gładkości tekstury
    # im wyższa wartość tym bardzie jednolita jest tekstura na obrazie
    # - homogeneity mierzy jak bardzo piksele w teksturze są podobne do siebie
    # im większa wartość tym bardziej podobną są pixele do siebie
    # - asm ocenia jednorodność tekstury na obrazie poprzez analizę występowania pikseli
    # dla różnych kombinacji wartości szarości
    features = calculate_texture_features(images, [1, 3, 5], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM'])  # oblicza cechy tekstury
    all_features.append(features)  # Dodaje cechy do listy cech
    labels.extend([os.path.basename(folder)] * len(features))  # Dodaje etykiety do listy etykiet

all_features = np.vstack(all_features)  # Łączy cechy wszystkich tekstur w jedną tablicę
df = pd.DataFrame(all_features)
df['label'] = labels
df.to_csv('texture_features.csv', index=True)

features_df = pd.read_csv('texture_features.csv', sep=',')

data = np.array(features_df)  # Konwertuje DataFrame na tablicę NumPy
X = data[:, :-1].astype('float64')  # Wyodrębnia cechy jako macierz X
Y = data[:, -1]  # Wyodrębnia etykiety jako wektor Y

classifier = svm.SVC(gamma='auto')  # Inicjalizuje klasyfikator SVM
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)  # Dzieli dane na zbiory treningowe i testowe

classifier.fit(X_train, Y_train)  # Dopasowuje model klasyfikatora do danych treningowych
Y_pred = classifier.predict(X_test)   # Dokonuje predykcji na danych testowych
accuracy = accuracy_score(Y_test, Y_pred)  # Oblicza dokładność klasyfikacji
print("Accuracy:", accuracy)
