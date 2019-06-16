#!/usr/bin/python3

import cv2
import numpy as np
import sys
import os
import pickle
from matplotlib import pyplot as plt
import datetime

N_MELHORES = 20
TAM_DIC = 50

# Funcao que retorna o nome da lista de imagens
def ler_lista_imagens(inp):
    lista = []
    with open(inp, "r") as f:
        for linha in f:
            lista.append(linha.strip())
            #print(linha.strip())
    return lista

# Funcao que retorna o nome da lista de imagens
def ler_diretorio_imagens(root_dir):
    lista = []
    root_dir = os.path.abspath(root_dir)
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in files:
            full_path = os.path.join(root, name)
            lista.append(full_path)
            # print(full_path)
    return lista

# Funcao que retorna os descritores da imagem de entrada
def extrair_descritores(inp, tecnica="todos"):
    img = cv2.imread(inp)
    if img is None:
        print("Imagem nao encontrada")
        return []

    if tecnica is "blur":
        # Reducao de noise
        # blur = cv2.GaussianBlur(img,(5,5),0)
        blur = cv2.bilateralFilter(img,9,75,75)
        processed = blur
    
    elif tecnica is "cinza":
        cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed = cinza

    elif tecnica is "canny":
        # Reconhecimento de bordas
        # rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # cinza = cv2.imread(file_name, 0)
        cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(cinza, 150,150)
        processed = canny
    
    elif tecnica is "todos":
        blur = cv2.bilateralFilter(img,9,75,75)
        cinza = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(cinza, 150,150)
        processed = canny

    else:
        return []

    # Extrator ORB (FAST keypoint detector + BRIEF descriptor)
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(processed, None)
    
    return kp, des

# Funcao que extrai todas as caracteristicas de uma lista de imagens
def extrair_lista_caracteristicas(imagens, label="-", tecnica="todos"):
    caracteristicas = []

    # Extrai todos os descritores e keypoints de cada imagem
    for imagem in imagens:
        kp, desc = extrair_descritores(imagem, tecnica)
        # print("\n", desc)

        caracteristicas.append({
            'nome': label,
            'descritores': desc,
            'path': imagem
        })
    
    return caracteristicas

# Funcao que compara a imagem base com uma lista e devolve
#  as mais similares dada uma nota de corte
def compara_caracteristicas(img_base, lista_imgs, nota_corte):

    desc_base = img_base['descritores']
    resultado = []

    media_total = 0

    for imagem in lista_imgs:
        
        desc = imagem['descritores']

        # Cria BFMatcher objeto
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Faz o matching de descritores
        matches = bf.match(desc_base, desc)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        count = 0
        value = 0

        for match in matches:
            if (count >= N_MELHORES):
                break
            value += match.distance
            count += 1

        # media = value/len(matches)
        media = value/N_MELHORES
        # print('Média:', media)

        if (media < nota_corte):
            imagem['score'] = media
            resultado.append(imagem)
        
        media_total += media
    
    print("Média total:", str(media_total/len(lista_imgs))[0:5])
    return resultado

# Extrai descritores dos 4 diretorios de monumentos
def extrai_caract_monumentos(tecnica="todos"):
    imagens = ler_diretorio_imagens("banco_imagens/Colosseum")
    Colosseum = extrair_lista_caracteristicas(imagens, "Colosseum", tecnica)

    imagens = ler_diretorio_imagens("banco_imagens/Eiffel")
    Eiffel = extrair_lista_caracteristicas(imagens, "Eiffel", tecnica)

    imagens = ler_diretorio_imagens("banco_imagens/Louvre")
    Louvre = extrair_lista_caracteristicas(imagens, "Louvre", tecnica)

    imagens = ler_diretorio_imagens("banco_imagens/Parthenon")
    Parthenon = extrair_lista_caracteristicas(imagens, "Parthenon", tecnica)

    return Colosseum + Eiffel + Louvre + Parthenon

# Carrega caracteristica do arquivo ou entao extrai na hora
def carregar_lista_caracteristicas():
    lista_caracteristicas = []
    try:
        lista_salva = open("lista_caract.pickle","rb")
        lista_caracteristicas = pickle.load(lista_salva)['lista']
    except Exception as e:
        print('EXCEPT:', e)
        
        lista_caracteristicas = extrai_caract_monumentos()

        lista_salva = open("lista_caract.pickle","wb")
        output = {
            'qtd_imagens': len(lista_caracteristicas),
            'lista': lista_caracteristicas
        }

        pickle.dump(output, lista_salva)
        lista_salva.close()
    return lista_caracteristicas

# Carrega uma imagem e seus descritores
def gerar_input(path_base = 'banco_imagens/Eiffel/dan-novac-1132798-unsplash.jpg', tecnica="todos"):
    # Imagem de base para comparacao
    kp_base, desc_base = extrair_descritores(path_base, tecnica)
    imagem_base = {
        'nome': "Não reconhecido",
        'descritores': desc_base,
        'path': path_base
    }
    return imagem_base

# Cria o dicionario de palavras visuais
def criar_dic(bag, tam_dicionario):
    bag = np.float32(bag)

    # Limita a 50 o num de tentativas do k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)

    ret, labels, center = cv2.kmeans(bag, tam_dicionario, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # retorna a lista de centroides de cada grupo de descritores
    return center

# Gera histograma a partir de um dicionario e um conjunto de descritores
def gerar_histograma(bovw, desc_base):
    hist = []
    for word in bovw:
        dist_total = 0
        for desc in desc_base:
            # calcular a distancia entre o descritor e o centro
            dist_total += cv2.norm(np.float32(desc), np.float32(word), cv2.NORM_L2)
        
        dist_media = dist_total/len(desc_base)
        hist.append(dist_media)

    return hist

# Carrega lista de histogramas de todas as imagens da base
def carrega_histogramas(bovw):
    lista_histogramas = []
    try:
        lista_salva = open("lista_histogramas.pickle","rb")
        lista_histogramas = pickle.load(lista_salva)
    except Exception as e:
        print('EXCEPT:', e)
        
        lista_carac = carregar_lista_caracteristicas()

        imagens = ler_diretorio_imagens("banco_imagens/Colosseum")
        imagens += ler_diretorio_imagens("banco_imagens/Eiffel")
        imagens += ler_diretorio_imagens("banco_imagens/Louvre")
        imagens += ler_diretorio_imagens("banco_imagens/Parthenon")

        for imagem in imagens:
            kp, desc = extrair_descritores(imagem)
            lista_histogramas.append(gerar_histograma(bovw, desc))

        lista_salva = open("lista_histogramas.pickle","wb")
        pickle.dump(lista_histogramas, lista_salva)
        lista_salva.close()
    return lista_histogramas

# Funcao Main
def main():
    # Extrai todos os descritores e coloca-os na mesma "bag"
    bag_features = []
    lista_caracteristicas = carregar_lista_caracteristicas()
    
    for item in lista_caracteristicas:
        desc = item['descritores']
        bag_features = bag_features + desc.tolist()
        # bag_features.append(desc)
    
    bovw = criar_dic(bag_features, TAM_DIC)

    imagem_base = gerar_input('banco_imagens/Colosseum/mathew-schwartz-629316-unsplash.jpg')
    hist_entrada = gerar_histograma(bovw, imagem_base['descritores'])

    histogramas = carrega_histogramas(bovw)

    imagens = ler_diretorio_imagens("banco_imagens/Colosseum")
    imagens += ler_diretorio_imagens("banco_imagens/Eiffel")
    imagens += ler_diretorio_imagens("banco_imagens/Louvre")
    imagens += ler_diretorio_imagens("banco_imagens/Parthenon")

    # Calcula as distâncias do histograma da imagem de entrada com as da lista de imagens...
    resultados = []
    for i in range(len(histogramas)):
        distancia = cv2.norm(np.float32(hist_entrada), np.float32(histogramas[i]), cv2.NORM_L2)
        resultados.append([distancia, imagens[i]])
        # print([imagens[i], distancia])
    
    resultados = sorted(resultados, key = lambda x:x[0])
    print('Imagem de entrada era:', str(resultados[0][1]).split('\\')[-2])
    for resultado in resultados[1:11]:
        print(str(resultado[0])[0:4], '|', str(resultado[1]).split('\\')[-2])

# # Funcao Main de testes
# def main_testes(tecnica="todos"):
#     lista_caracteristicas = carregar_lista_caracteristicas()
#     imagem_base = gerar_input('banco_imagens/Colosseum/mathew-schwartz-629316-unsplash.jpg')

#     similares = compara_caracteristicas(
#         imagem_base, lista_caracteristicas, 100)
    
#     similares = sorted(similares, key = lambda x:x['score'])
#     for similar in similares[0:10]:
#         print('[' + str(similar['score'])[0:4] + '] ' + similar['nome'] +
#             ' | ' + similar['path'].split('\\')[-1])


if __name__ == "__main__":
    start = datetime.datetime.now()

    # main_testes()
    main()

    end = datetime.datetime.now()
    print('Tempo de processamento:', str(end-start).split('.')[0])
