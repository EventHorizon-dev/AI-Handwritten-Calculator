import keras
import numpy as np
import os
import glob
from PIL import Image
import kagglehub


print("Téléchargement du dataset des symboles...")
path_dataset = kagglehub.dataset_download("xainano/handwrittenmathsymbols")

# On pointe directement vers le dossier qui contient tous les symboles
path_dataset = os.path.join(path_dataset, "extracted_images")




def charger_images_dossier(nom_dossier, label_class, max_samples=4000):
    images=[]
    labels=[]

    # Recherche du sous-dossier correspondant dans le dataset téléchargé
    dossier_cible=os.path.join(path_dataset,nom_dossier)

    #Si le fichier n'est pas trouvé dans le dossier principal, on cherche dans les sous-dossiers avec glob.
    if not os.path.exists(dossier_cible):
        recherches=glob.glob(os.path.join(path_dataset,"**", nom_dossier),recursive=True)
        if recherches:
            dossier_cible=recherches[0]
        else:
            print(f"Dossier introuvable: {nom_dossier}")
            #Si on ne trouve vraiment rien, on renvoi des tableaux vides pour ne pas faire planter le programme.
            return np.empty((0,28,28)), np.empty((0,))

    #Ne garder que les fichiers .png, .jpg, .jpeg et couper la liste pour ne garder que le nombre maximal d'image autorisées
    fichiers=[f for f in os.listdir(dossier_cible) if f.endswith((".png",".jpg", ".jpeg"))]
    fichiers=fichiers[:max_samples]

#On parcours tous les fichier de la liste et pour chacun on recrée le chemin jusqu'au fichier d'image à partir de son nom:
    # On convertit les images en niveau de gris et on les resize pour qu'elle fasse 28 par 28 pixels comme les images MNIST

    for filename in fichiers:
        img_path = os.path.join(dossier_cible, filename)
        img=Image.open(img_path).convert('L').resize((28,28))
        img_array = np.array(img)
#On calcule la luminosité avec np.mean et si elle est supérieur à 127 c'est que l'image est sur du noir sur fond blanc et alors on la convertit en blanc sur fond noir(comme MNIST) pour éviter de perturber L'IA
        if np.mean(img_array)>127:
            img_array=255-img_array

        images.append(img_array)
        labels.append(label_class)

    return np.array(images), np.array(labels)



modele=keras.Sequential()

#Definition des 3 couches
modele.add(keras.layers.InputLayer(input_shape=(784,)))
modele.add(keras.layers.Dense(64, activation='relu'))
modele.add(keras.layers.Dense(12, activation='softmax'))

#adam:permet à l'IA d'ajuster ses poids pour faire de moins en moins d'erreurs
#loss='sparse_categorical_crossentropy': mesure à quel point la réponse de L'IA est éloignée de la réalité
#metrics=['accuracy']: permet de suivre le pourcentage de bonne réponse de l'IA
modele.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

#importer les données de MNIST:
#X_train et y_train: images d'entraînement
#X_test et y_test: images de test
(X_train, y_train), (X_test, y_test) =keras.datasets.mnist.load_data()

#Charger les 3 opérateurs et leur attribuer leur numéro de neurone
X_plus, y_plus=charger_images_dossier('+', label_class=10)
X_moins, y_moins=charger_images_dossier('-', label_class=11)




X_train = np.concatenate([X_train, X_plus, X_moins], axis=0)
y_train = np.concatenate([y_train, y_plus, y_moins], axis=0)


#On passe les pixels de 0-255 à 0.0-1.0 pour rendre les calculs plus faciles
X_train = X_train / 255.0
X_test = X_test / 255.0
#Aplatissement des images (on passe de (28, 28) à 784 pixels en ligne)
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

#Entraînement de l'IA:
#epochs: c'est le nombre de fois que L'IA va voir les images(plus elle les revoit, mieux elle apprend.
#batch_size=32 : C'est le nombre d'images que l'IA analyse en même temps avant de corriger un petit peu ses erreurs.
modele.fit(X_train, y_train, epochs=5, batch_size=32)

modele.save('mon_modele_IA_keras_V2.keras')
print("Modèle V2 entraîné et sauvegardé avec succès !")