import keras
import tkinter as tk
from PIL import Image, ImageOps, ImageDraw
import numpy as np
from markdown_it.rules_block import table

modele=keras.models.load_model('mon_modele_IA_keras_V2.keras')

class AppCalculatriceIA:
    def __init__(self, fenetre_principale):
        self.fenetre=fenetre_principale
        self.fenetre.title("Calculatrice IA")

        #On associe notre modèle chargé au moteur
        self.modele=modele

        #Création du dictionnaire de stockage du calcul
        self.elements_equation=[]

        # Création de l'image PIL en mémoire pour enregistrer le dessin (en niveaux de gris "L")
        self.image_courante_pil = Image.new("L", (200, 200), color=255)
        self.dessin = ImageDraw.Draw(self.image_courante_pil)

        #Création de la fenêtre de dessin interactive
        self.canvas = tk.Canvas(self.fenetre, width=200, height=200, bg="white")
        self.canvas.pack(pady=10)

        #Lier le glissement de la souris à la fonction de dessin
        self.canvas.bind("<B1-Motion>", self.dessiner)

        self.btn_valider = tk.Button(self.fenetre, text="Valider", command=self.valider_dessiner, bg="green", fg="white")
        self.btn_valider.pack(side=tk.LEFT, padx=20, pady=10)

        self.btn_effacer = tk.Button(self.fenetre, text="Effacer", command=self.effacer_canvas, bg="red", fg="white")
        self.btn_effacer.pack(side=tk.RIGHT, padx=20, pady=10)
        self.btn_calculer=tk.Button(self.fenetre, text="Calculer", command=self.calculer_equation, bg="blue", fg="white")
        self.btn_calculer.pack(side=tk.LEFT, padx=20, pady=10)

        #---Système de correction---

        self.label_correction=tk.Label(self.fenetre, text="CORRECTION")
        self.label_correction.pack(pady=5)

        #Cadre pour les chiffres (0 à 9)
        cadre_chiffres=tk.Frame(self.fenetre)
        cadre_chiffres.pack()
        for i in range(10):
            btn=tk.Button(cadre_chiffres,text=str(i),width=2,command=lambda c=i:self.corriger_ia(c))
            btn.pack(side=tk.LEFT, padx=1)
        #Cadre pour les opérateurs(+ et -)
        cadre_operateur=tk.Frame(self.fenetre)
        cadre_operateur.pack(pady=3)
        btn_plus=tk.Button(cadre_operateur, text="+", width=4, bg="lightgray",command=lambda:self.corriger_ia(10))
        btn_plus.pack(side=tk.LEFT, padx=2)
        btn_moins=tk.Button(cadre_operateur,text="-",width=4, bg="lightgray",command=lambda:self.corriger_ia(11))
        btn_moins.pack(side=tk.LEFT, padx=2)
    def dessiner(self, event):
        x,y=event.x,event.y
        r=8 # Le rayon du pinceau
        # Dessiner sur le Canvas visible (en noir sur fond blanc)
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        # Dessiner en même temps sur l'image PIL en mémoire (pour l'IA)
        self.dessin.ellipse([x-r, y-r, x+r, y+r], fill=0)
        #Boutons de contrôle(Valider et effacer)


    def effacer_canvas(self):
        self.canvas.delete("all")
        self.image_courante_pil = Image.new("L", (200, 200), color=255)
        self.dessin = ImageDraw.Draw(self.image_courante_pil)

    def corriger_ia(self,vraie_classe):
        #Vérifier qu'il y a bien un dessin en mémoire
        if hasattr(self,"dernier_dessin")and self.dernier_dessin is not None:
            vraie_etiquette=np.array([vraie_classe])
            self.modele.train_on_batch(self.dernier_dessin, vraie_etiquette)
            self.modele.save('mon_modele_IA_keras_V2.keras')
            if vraie_classe <= 9:
                nom_symbole = str(vraie_classe)
            elif vraie_classe == 10:
                nom_symbole = '+'
            elif vraie_classe == 11:
                nom_symbole = '-'
            else:
                nom_symbole = '?'
            if self.elements_equation:
                self.elements_equation[-1] = nom_symbole
            print(f"Correction enregistrée ! L'IA a appris que ce dessin est : {nom_symbole}")
            print("Equation actuelle:", " ".join(self.elements_equation))

            self.dernier_dessin=None

        else:
            print("Dessine d'abord et clique sur Valider avant de corriger !")

            # : On teste tout de suite si l'IA a compris la leçon
            prediction = self.modele.predict(self.dernier_dessin)
            nouvelle_classe = np.argmax(prediction)
            print("Nouvelle prédiction de l'IA après correction (classe) :", nouvelle_classe)

    def valider_dessiner(self):
        #Redimensionner l'image
        img_redimensionnee=self.image_courante_pil.resize((28,28))

        #Inverser les couleurs car MNIST attend du blanc sur fond noir
        img_inverse=ImageOps.invert(img_redimensionnee)

        #Convertir en tableau NumPy et normaliser(diviser par 255.0)
        tableau=np.array(img_inverse)/255.0

        # Aplatir le tableau en 784 pixels pour correspondre à la couche d'entrée
        tableau = tableau.reshape(1, 784)
        self.dernier_dessin=tableau

        #Demander à l'IA de faire sa prédiction
        #argmax permet de prendre la plus grande probabilité de la liste renvoyée par l'IA
        prediction=self.modele.predict(tableau)
        classe_predite=np.argmax(prediction)

        #Traduction de la classe en caractère lisisbles
        if 0<=classe_predite<=9:
            symbole=str(classe_predite)
        elif classe_predite==10:
            symbole="+"
        elif classe_predite==11:
            symbole="-"
        else:
            symbole="?"
        print(f"L'IA a reconnu : {symbole} (classe {classe_predite})")
        self.elements_equation.append(symbole)
        print("Équation actuelle :", " ".join(self.elements_equation))

    def calculer_equation(self):
        if not self.elements_equation:
            print("Aucune équation à calculer")
            return

        equation_texte="".join(self.elements_equation)
        print(f"Equation complète : {equation_texte}")

        try:
            resultat=eval(equation_texte)
            print(f"Résultat : {resultat}")
            self.elements_equation.clear()
            self.effacer_canvas()

        except Exception as e:
            print("Syntaxe de l'équation incorrecte, impossible de la calculer")

if __name__ == "__main__":
    fenetre = tk.Tk()
    app = AppCalculatriceIA(fenetre)
    fenetre.mainloop()

