import os
from PIL import Image


def cut_textures(import_dir, output_dir, size=(128, 128)):
    # Sprawdzenie czy katalog wyjściowy istnieje
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Pętla po wszystkich plikach w katalogu wejściowym
    for filename in os.listdir(import_dir):
        # Ścieżka do aktualnego pliku
        file_path = os.path.join(import_dir, filename)

        # Sprawdzenie czy to plik graficzny
        if os.path.isfile(file_path) and any(file_path.endswith(extension) for extension in ['.jpg', '.png', '.jpeg']):
            # Wczytanie obrazu
            image = Image.open(file_path)

            # Pętla po wszytkich możliwych pozycjach wycięcia
            # pozycje wycięcia odnoszą się do współrzędnych na onrazie, z których wycinane są fragmenty 128 x 128
            # każdy fragment zaczyna się od określonej pozycji x, y na obrazie i kończy się na pozycji
            # x + szerokość frag., y + wysokość frag.
            # iterowanie przez cały obraz
            for y in range(0, image.height - size[1] + 1, size[1]):
                for x in range(0, image.width - size[0] + 1, size[0]):
                    # Wycięcie fragmentu
                    texture = image.crop((x, y, x + size[0], y + size[1]))
                    output_filename = f"{os.path.splitext(filename)[0]}_{y}_{x}.jpg"
                    output_subdir = os.path.join(output_dir, os.path.splitext(filename)[0])

                    # Sprawdzenie czy katalog wyjściowy dla danego obrazu istnieje
                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)

                    # Zapis fragmentu do pliku
                    texture.save(os.path.join(output_subdir, output_filename))

input_directory = "C:\\Studia 2\\Semestr 1\\Programowanie w obliczeniach inteligentnych\\Laboratorium 3"
output_directory = "C:\\Studia 2\\Semestr 1\\Programowanie w obliczeniach inteligentnych\\Laboratorium 3\\out"
texture_size = (128, 128)
cut_textures(input_directory, output_directory, size=texture_size)