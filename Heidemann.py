# -*- coding: utf-8 -*-
"""
Created on Wednesday Mars  7  2018

@author: Mannaig L'Haridon et Habib Doucouré
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from Reisfeld import derivees, gradient, theta, gamma

"""
#############################################################################
#                                                                           #
#                   TP : EXTRACTION DE POINTS D'INTERETS                    #
#     Partie II : HEIDEMANN (traitement des images en niveaux de gris)      #
#                                                                           #
#############################################################################
"""

"""
fonction gamma R à modifier ! :) calculer des delta !
"""

####################  CALCUL DE LA CARTE DE SYMETRIE  ####################


"""
Rajouter les k et l dans les fonctions ! (issus de delta)
"""


def GWF(G_I,pi,pj):
    """
    Gradient Weight Function : Pondération de la paire de pixels
    """
    gwf = np.log(1+G_I[pi[0],pi[1]])*np.log(1+G_I[pj[0],pj[1]])
    return gwf


def PWF(Theta_I,pi,pj):
    """
    Phase Weight Function
    """
    if pi[1]==0:
        alpha = 0
    else:
        alpha = np.arctan(pi[0]/pi[1])
    gamma_i = Theta_I[pi[0]][pi[1]]-alpha
    gamma_j = Theta_I[pj[0],pj[1]]-alpha
    pwf_moins = 1-np.cos(gamma_i-gamma_j) 
    pwf_plus = 1-np.cos(gamma_i+gamma_j) 
    pwf = pwf_plus*pwf_moins
    return pwf


def symetrie(l,c,gamma_R,G_I,Theta_I):
    """
    Valeur de la symetrie pour un pixel
    """
    s = 0
    taille = int(np.size(gamma_R)/2)
    for i in range(taille):
        if i%2 ==0:
            # Translation d'un vecteur (l,c)
            pi = np.add(gamma_R[i],[l,c])
            pj = np.add(gamma_R[i+1],[l,c])
            s += PWF(Theta_I,pi,pj) * GWF(G_I,pi,pj)
    return s



#################### DETERMINATION DES POINTS D'INTERETS ####################        

def convGauss(carteSym,sigma):
    """
    Convolution de la carte de symetrie par une gaussienne
    """
    conv = gaussian_filter(carteSym,sigma)
    return conv


def seuil_loc(max_loc,conv):
    """
    Détermination du seuil moyen des maxima locaux
    """
    taille = max_loc.shape[0]
    seuil = 0
    for t in range(taille):
        seuil += conv[max_loc[t][0]][max_loc[t][1]]
    seuil /= taille
    return seuil


def maxLocaux(max_loc,conv,seuil,fichier):
    """
    Détection des maximum locaux supérieurs au seuil
    """
    maxL = []
    maxC = []
    taille = max_loc.shape[0]
    for t in range(taille):
        if conv[max_loc[t][0]][max_loc[t][1]]>seuil:
            maxL.append(max_loc[t][0])
            maxC.append(max_loc[t][1])
            fichier.write(str(max_loc[t][1])+" "+str(max_loc[t][0])+"\n")
    MaxL = np.asarray(maxL)
    MaxC = np.asarray(maxC)
    return MaxL,MaxC





if __name__ == "__main__" :
    
    # Lecture de l'image à étudier
    I = io.imread('dalleRVB.tif')
    I = I/256
    plt.figure()
    plt.title("Image brute")
    plt.imshow(I)

    # Paramètres
    L,C,canaux = I.shape
    R = 3
    sigma = 1
    sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    # Separation de l'image I en 3 images d'un seul canal
    Ir = np.zeros((L,C))
    Iv = np.zeros((L,C))
    Ib = np.zeros((L,C))
    for l in range(L):
        for c in range(C):
            Ir[l][c] = I[l][c][0]
            Iv[l][c] = I[l][c][1]
            Ib[l][c] = I[l][c][2]
    
    # Dérivées de l'image calculées avec le filtre de Sobel
    Ix_r, Iy_r = derivees(Ir,sobel)
    Ix_v, Iy_v = derivees(Iv,sobel)
    Ix_b, Iy_b = derivees(Ib,sobel)
    
    # Calcul du module des gradients
    G_Ir = gradient(Ix_r,Iy_r)
    G_Iv = gradient(Ix_v,Iy_v)
    G_Ib = gradient(Ix_b,Iy_b)
    plt.figure()
    plt.suptitle("Module du gradient")
    plt.subplot(1,3,1)
    plt.title("Canal rouge")
    plt.imshow(G_Ir)
    plt.subplot(1,3,2)
    plt.title("Canal vert")
    plt.imshow(G_Iv)
    plt.subplot(1,3,3)
    plt.title("Canal bleu")
    plt.imshow(G_Ib)
    
    # Calcul de la direction des gradients
    Theta_Ir = theta(Ix_r,Iy_r)
    Theta_Iv = theta(Ix_v,Iy_v)
    Theta_Ib = theta(Ix_b,Iy_b)
    plt.figure()
    plt.suptitle("Direction du gradient")
    plt.subplot(1,3,1)
    plt.title("Canal rouge")
    plt.imshow(Theta_Ir)
    plt.subplot(1,3,2)
    plt.title("Canal vert")
    plt.imshow(Theta_Iv)
    plt.subplot(1,3,3)
    plt.title("Canal bleu")
    plt.imshow(Theta_Ib)
    
    gamma_R = gamma(R)  #Pixels se trouvant dans un rayon R à un pixel de ref.
    
    
    # Calcul de la carte de symetrie
    S = np.zeros((L,C,3))
    for l in range(R,L-R+1):
        for c in range(R,C-R+1):
            S[l][c][0] = symetrie(l,c,gamma_R,G_Ir,Theta_Ir)
            S[l][c][1] = symetrie(l,c,gamma_R,G_Iv,Theta_Iv)
            S[l][c][2] = symetrie(l,c,gamma_R,G_Ib,Theta_Ib)
    plt.figure()
    plt.title("Carte de symétrie")
    plt.imshow(S)
    
    # Creation du fichier d'enregistrement des points d'interets
    fichier = open("pointsInterets_rvb.txt","w")
    
    # Detection et enregistrement des points d'interets        
    conv = convGauss(S,sigma)
    plt.figure()
    plt.title("Image convoluée")
    plt.imshow(conv)
    
    conv_r = np.zeros((L,C))
    conv_v = np.zeros((L,C))
    conv_b = np.zeros((L,C))
    for l in range(L):
        for c in range(C):
            conv_r = conv[l][c][0]
            conv_v = conv[l][c][1]
            conv_b = conv[l][c][2]

    max_loc_r = peak_local_max(conv_r,min_distance=10)
    max_loc_v = peak_local_max(conv_v,min_distance=10)            
    max_loc_b = peak_local_max(conv_b,min_distance=10)
#    seuil = seuil_loc(max_loc,conv)
#    maxL, maxC = maxLocaux(max_loc,conv,seuil,fichier)
#    
#    plt.figure()
#    plt.title("Points d'intérêts de l'image")
#    plt.imshow(I)
#    plt.plot(maxC,maxL,'b+')
#    plt.show()
    
    
    
    
    
    