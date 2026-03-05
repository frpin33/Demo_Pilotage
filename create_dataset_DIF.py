import os ,random, pickle , tifffile , shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from osgeo import gdal, ogr



def pyplot() :
    img_folder = 'U:/CarteAssistéEssence/transfert_techno_classif_essences/notebook/Datasets/train/'
    

    images = [os.path.join(img_folder,i) for i in os.listdir(img_folder) if i.endswith(".tif")]

    # Sélection des images à afficher
    n_to_show = 2
    idx_to_show = random.sample(range(len(images)), n_to_show)

    # Lecture des images et des essences correspondates
    rgb, var, ess = [], [], []
    for i in idx_to_show:
        data = tifffile.imread(images[i])
        print(np.random.normal(0, 0.05, data.shape))
        rgb.append(data[:,:,:3])
        var.append(data[:,:,5])
        ess.append(data[:,:,9])
        dataloop = np.copy(data)
        img = dataloop[:,:,:4]
        img = (img + np.random.normal(0, 0.05, img.shape)).astype(np.float32)
        dataloop[:,:,:4] = img
        rgb.append(dataloop[:,:,1:4])
        var.append(dataloop[:,:,5])
        ess.append(dataloop[:,:,9])

        aa = data[:,:,4]
        ab = data[:,:,5]
        ac = data[:,:,6]
        ada = data[:,:,7]
        aea = data[:,:,8]
        afa = data[:,:,9]
        aga = data[:,:,10]
        aha = data[:,:,11]


    # Création de la figure
    f, ax = plt.subplots(4,3)
    # Définir la taille d'image 
    f.set_figheight(20)
    f.set_figwidth(20)
    # Affichage des images
    for i in range(4):
        ax[i,0].imshow(rgb[i])
        ax[i,1].imshow(var[i], clim=[0,1])
        ax[i,2].imshow(ess[i], clim=[0,1])

    # Montrer les statistiques du dataset
    plt.show()
    print('Taille des imagettes : ', data.shape)
    print('Valeur minimale : ', np.min(data[:,:,:-1]))
    print('Valeur maximale : ', np.max(data[:,:,:-1])) 

pyplot()
