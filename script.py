import os
import kagglehub
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# --- 1. DOWNLOAD DO DATASET VIA KAGGLEHUB ---
print("Verificando/Baixando dataset...")
path_root = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
print(f"Arquivos localizados em: {path_root}")

# --- 2. CONFIGURAÇÕES DE CAMINHO ---
# O dataset da Andrada geralmente tem a estrutura: pasta_baixada/Data/genres_original
DATASET_PATH = os.path.join(path_root, 'Data', 'genres_original')

# Verificação de segurança: se a pasta não existir nesse caminho, tenta o padrão
if not os.path.exists(DATASET_PATH):
    # Tenta procurar apenas 'genres' caso a estrutura mude
    DATASET_PATH = os.path.join(path_root, 'genres')

if not os.path.exists(DATASET_PATH):
    raise Exception(f"Não foi possível encontrar a pasta de gêneros dentro de {path_root}. Verifique a estrutura.")

print(f"Lendo áudios de: {DATASET_PATH}")

# Onde as imagens serão salvas (na pasta atual do seu projeto)
OUTPUT_PATH = './spectrograms_img'

# Configurações de Áudio
SAMPLE_RATE = 22050
TRACK_DURATION = 30 
SLICES_PER_TRACK = 10 
SLICE_DURATION = TRACK_DURATION // SLICES_PER_TRACK

def generate_spectrograms():
    # Cria a pasta de saída se não existir
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # Loop por todos os gêneros (pastas)
    genres = os.listdir(DATASET_PATH)
    
    for genre in genres:
        genre_path = os.path.join(DATASET_PATH, genre)
        
        # Verifica se é uma pasta válida
        if not os.path.isdir(genre_path):
            continue
            
        print(f"Processando gênero: {genre}...")
        
        # Cria subpasta para o gênero nas imagens de saída
        output_genre_path = os.path.join(OUTPUT_PATH, genre)
        os.makedirs(output_genre_path, exist_ok=True)
        
        # Loop por todos os arquivos de áudio
        files = os.listdir(genre_path)
        
        for filename in files:
            # Ignora arquivos que não sejam de áudio (às vezes vem arquivos .csv junto)
            if not filename.endswith('.wav'):
                continue

            file_path = os.path.join(genre_path, filename)
            
            try:
                # Carrega o áudio
                y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                samples_per_slice = int(SAMPLE_RATE * SLICE_DURATION)
                
                for i in range(SLICES_PER_TRACK):
                    start_sample = samples_per_slice * i
                    end_sample = start_sample + samples_per_slice
                    
                    segment = y[start_sample:end_sample]
                    
                    if len(segment) != samples_per_slice:
                        continue
                    
                    # Gera o Espectrograma
                    melspectrogram = librosa.feature.melspectrogram(y=segment, sr=sr)
                    melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)
                    
                    # Salva a Imagem
                    save_name = f"{filename[:-4]}_slice{i}.png"
                    save_path = os.path.join(output_genre_path, save_name)
                    
                    plt.imsave(save_path, melspectrogram_db, origin='lower', cmap='magma')
                    
            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")

    print("\nConcluído! Imagens geradas em:", os.path.abspath(OUTPUT_PATH))

if __name__ == "__main__":
    generate_spectrograms()