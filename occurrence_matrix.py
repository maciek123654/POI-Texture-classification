import os
import numpy as np
from skimage import io, color, feature
from skimage.util import img_as_ubyte
from itertools import product
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt


def occurrence_matrix(image_path):
    # Wczytanie obrazu
    image = io.imread(image_path)
    # Konwersja obrazu do skali szarości
    gray_image = color.rgb2gray(image)
    # Zmniejszenie głębi ostrości do 5 bitów
    # zmniejsza to złożonośc obliczeniową oraz poprawia wpływ szumów
    qu_image = img_as_ubyte(gray_image // 64 * 64)
    # Definicja odległości i kierunków
    # małe odległości - 1px, znajdowanie subtelnych różnic
    # większe odległości - 5px,  analiza dużych obszarów
    # 0 stopni - 0
    # 45 stopni - np.pi/4
    # 90 stopni - np.pi/2
    # 135 stopni - 3*np.pl/4, uzględnianie poziomych, pionowych i ukośnych tekstur
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    # wyznaczanie cech tekstury dla każdej kombinacji odległości i kierunku
    features = []
    for distance, angle in product(distances, angles):
        # macierz zdarzeń dla zadanego obrazu kwantyzowanego
        glcm = graycomatrix(qu_image, distances=[distance], angles=[angle], symmetric=True, normed=True)
        # dissimilarity odnosi się do odmienności tekstury na obrazie
        # mierzy to średnią różnicę w intensywności między pikselami w obrazie
        # im wyższa wartość tym bardziej zróżnicowana jest tekstura
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        # correlation odnosi się do korelacji między intensywnościami pikseli w obrazie
        # im wyższa wartość tym bardziej liniowo zależne są piksele w teksturze
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        # contrast odnosi się do kontrastu między intensywnoścami pikseli w analizowanej teksturze
        # im wyższa wartość tym większa różnica w intensywności pikseli i większy kontrast tekstury na obrazie
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        # energy mierzy sumę kwadratów wartości pikseli w macierzy
        # jest to miara jednorodności lub gładkości tekstury
        # im wyższa wartość tym bardzie jednolita jest tekstura na obrazie
        energy = graycoprops(glcm, 'energy')[0, 0]
        # homogeneity mierzy jak bardzo piksele w teksturze są podobne do siebie
        # im większa wartość tym bardziej podobną są pixele do siebie
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        # asm ocenia jednorodność tekstury na obrazie poprzez analizę występowania pikseli dla różnych kombinacji wartości szarości
        asm = graycoprops(glcm, 'ASM')[0, 0]

        # Uzupełnianie wektora cech o nazwę kategorii tekstury, odległość i kierunek
        category = f"Distance_{distance}_Angle_{int(np.degrees(angle))}_Degrees"
        texture_features = [dissimilarity, correlation, contrast, energy, homogeneity, asm, category]
        features.append(texture_features)
    return features


image_path = "C:\\Studia 2\\Semestr 1\\Programowanie w obliczeniach inteligentnych\\Laboratorium 3\\test.jpg"
texture_features = occurrence_matrix(image_path)
print("Wektor cech tekstury:")
for features in texture_features:
    print(features)
