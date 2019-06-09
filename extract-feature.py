#!/usr/bin/python3

import cv2
import numpy as np
import sys
import os
import pickle
from matplotlib import pyplot as plt

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
def extrair_descritores(inp):
    img = cv2.imread(inp)
    if img is None:
        print("Imagem nao encontrada")
        return []

    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extrator ORB (FAST keypoint detector + BRIEF descriptor)
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(cinza, None)
    
    return kp, des

# Funcao que extrai todas as caracteristicas de uma lista de imagens
def extrair_lista_caracteristicas(imagens):
    caracteristicas = []

    # Extrai todos os descritores e keypoints de cada imagem
    for imagem in imagens:
        kp, desc = extrair_descritores(imagem)
        # print("\n", desc)

        caracteristicas.append({
            'descritores': desc,
            'keypoints': kp,
            'path': imagem
        })
    
    return caracteristicas

# Funcao que compara a imagem base com uma lista e devolve
#  as mais similares dada uma nota de corte
def compara_caracteristicas(img_base, lista_imgs, nota_corte):

    desc_base = img_base['descritores']
    resultado = []

    for imagem in lista_imgs:
        
        desc = imagem['descritores']

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(desc_base, desc)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        count = 0
        value = 0
        for match in matches:
            if (count >= 15):
                break

            # print(match.distance)
            value += match.distance
            count += 1

        media = value/15
        # print('Média:', media)

        if (media < nota_corte):
            resultado.append(imagem)
    
    return resultado

# Funcao Main
def main():
    
    imagens = ler_diretorio_imagens("banco_imagens")
    
    # Para fins de teste, esta pegando a primeira imagem
    #  do proprio banco para usar como entrada
    path_base = 'banco_imagens/cristo1.jpg'

    # Imagem de base para comparacao
    kp_base, desc_base = extrair_descritores(path_base)
    imagem_base = {
        'descritores': desc_base,
        'keypoints': kp_base,
        'path': path_base
    }

    # 
    # TODO: tentar salvar o vetor de caracteristicas,
    #  para nao ter que processar todas as vezes
    # 
    # try:
    #     lista_salva = open("lista_caract.pickle","rb")
    #     lista_caracteristicas = pickle.load(lista_salva)
    # except Exception as e:
    #     lista_caracteristicas = extrair_lista_caracteristicas(imagens)
    #     lista_salva = open("lista_caract.pickle","wb")
    #     output = {
    #         'qtd_imagens': len(lista_caracteristicas),
    #         'lista': lista_caracteristicas
    #     }
    #     pickle.dump(output, lista_salva)
    #     lista_salva.close()

    lista_caracteristicas = extrair_lista_caracteristicas(imagens)

    similares = compara_caracteristicas(
        imagem_base, lista_caracteristicas, 40)
    
    # Printa o nome das imagens consideradas 
    #  similares com a entrada
    for similar in similares:
        print(similar['path'].split('\\')[-1])


if __name__ == "__main__":
    main()
