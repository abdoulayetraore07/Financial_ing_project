import math as mt
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.linalg import cholesky




def plot_graph_trajet(nb_simul_list, barriere,  prices, lower_bounds, upper_bounds, prix_theorique = 0):
 
    # Tracer les résultats
    plt.figure(figsize=(10, 6))
    plt.plot(nb_simul_list, prices, label="Prix estimé de l'option", color="blue", marker='o')

    # Convertir en arrays NumPy pour les opérations
    prices_array = np.array(prices)
    lower_bounds_array = np.array(lower_bounds)
    upper_bounds_array = np.array(upper_bounds)
    
    # Calculer les barres d'erreur
    yerr_lower = prices_array - lower_bounds_array
    yerr_upper = upper_bounds_array - prices_array
    
    # Tracer les points avec les barres d'erreur
    plt.errorbar(nb_simul_list, prices,
                yerr=[yerr_lower, yerr_upper],  # Barres d'erreur asymétriques
                fmt='o', color='blue', 
                capsize=3,      # Réduire de 5 à 3 pour des caps plus petits
                capthick=0.5,   # Réduire de 1 à 0.5 pour des caps plus fins
                ecolor='green',  
                elinewidth=0.7, # Paramètre pour réduire l'épaisseur des lignes
                alpha=0.7,      # Transparence (0.5 = 50% transparent)
                label="Intervalle de confiance")

    # Tracer les intervalles de confiance
    plt.fill_between(nb_simul_list, lower_bounds, upper_bounds, color="blue", alpha=0.1, label="Intervalle de confiance 90%")

     # Ajouter une ligne horizontale pour le prix théorique
    if not barriere :
        plt.axhline(y=prix_theorique, color='red', linestyle='--', label=f"Prix théorique : {prix_theorique:.4f}")
        plt.title("Estimation du prix d'une option européenne vanille avec intervalle de confiance")
        plt.xlabel("Nombre de simulations")
        plt.ylabel("Prix estimé de l'option")
        print(f"Le prix théorique de l'option est : {prix_theorique:.4f}\n")
    else :
        # Ajouter des détails au graphique
        plt.title("Estimation du prix d'une option à barrière Down & Out avec intervalle de confiance")
        plt.xlabel("Nombre de simulations")
        plt.ylabel("Prix estimé de l'option")

    plt.legend()
    plt.grid(True)
    plt.show()





def fonction_repart(x):
   # Constantes de la méthode d'Abramowitz et Stegun
    b_O = 0.2316419
    b_1 = 0.3193811530
    b_2 = -0.356563782
    b_3 = 1.781477937
    b_4 = -1.821255978
    b_5 = 1.330274429
    
    # Calcul du terme t
    t = 1 / (1 + b_O * x)
    
    # Calcul de la fonction de répartition (C.D.F.) de la loi normale standard
    poly = b_1 * t + b_2 * t**2 + b_3 * t**3 + b_4 * t**4 + b_5 * t**5
    FDR = 1 - 1 / mt.sqrt(2 * mt.pi) * mt.exp(-x**2 / 2) * poly

    return FDR




def simuler_W(nb_simul, T=2, barriere = False, delta = 1/52):
    """
    Simulation de mouvements browniens
    
    Args:
        nb_simul (int): Nombre de simulations
        T (float): Intervalle de temps
    
    Returns:
        numpy.ndarray: Valeurs simulées du mouvement brownien
    """
    if not barriere:
        gaussien_w = []
        for _ in range(nb_simul // 2):
            U_1 = random.uniform(0.0, 1.0)
            U_2 = random.uniform(0.0, 1.0)
            W_1 = mt.sqrt(-2 * mt.log(U_1)) * mt.cos(2 * mt.pi * U_2)
            W_2 = mt.sqrt(-2 * mt.log(U_1)) * mt.sin(2 * mt.pi * U_2)

            gaussien_w.append(W_1)
            gaussien_w.append(W_2)
            
        return np.array(gaussien_w)
    
    else:
        N_delta = mt.floor(T/delta)
        T = [(i+1)*delta for i in range(N_delta)] 

        G = np.random.normal(0,1,(N_delta,nb_simul))
        Gamma = np.zeros((N_delta, N_delta))
        I = np.zeros((N_delta, N_delta))

        for i in range(N_delta):
            Gamma[i, i:N_delta] = T[i]
            I[i,i] =  T[i]
            
        Gamma = Gamma + Gamma.T  - I
        A = cholesky(Gamma)
        gaussien_w = np.dot(A,G)

        return gaussien_w.T
    




def plot_graph(resultats):
    """
    Tracer la distribution des nombres gaussiens
    
    Args:
        resultats (numpy.ndarray): Données à visualiser
    """
    plt.figure(figsize=(10, 6))
    
    # Histogramme
    plt.hist(resultats, bins=300, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Courbe de densité de probabilité gaussienne
    x = np.linspace(min(resultats), max(resultats), 100)
    plt.plot(x, 1/(np.sqrt(2 * np.pi)) * np.exp( - (x)**2 / 2), 
             linewidth=2, color='red', label='Distribution Normale Théorique')
    
    plt.title('Distribution des Nombres Gaussiens')
    plt.xlabel('Valeurs')
    plt.ylabel('Densité de Probabilité')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
   
    # Affichage de la figure
    plt.show()