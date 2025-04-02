import math as mt
import random
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt



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



def simuler_W(nb_simul, T=2):
    """
    Simulation de mouvements browniens
    
    Args:
        nb_simul (int): Nombre de simulations
        T (float): Intervalle de temps
    
    Returns:
        numpy.ndarray: Valeurs simulées du mouvement brownien
    """
    gaussien_w = []
    for _ in range(nb_simul // 2):
        U_1 = random.uniform(0.0, 1.0)
        U_2 = random.uniform(0.0, 1.0)
        W_1 = mt.sqrt(-2 * mt.log(U_1)) * mt.cos(2 * mt.pi * U_2)
        W_2 = mt.sqrt(-2 * mt.log(U_1)) * mt.sin(2 * mt.pi * U_2)

        gaussien_w.append(W_1)
        gaussien_w.append(W_2)
    return mt.sqrt(T)*np.array(gaussien_w)



def calcul_P_euro(nb_simul, barriere=False, So=1, r=0.015, T=2, sigma=0.15, K=1, alph = 0.1):
    
    """
    Calcul du prix d'une option européenne PUT avec intervalle de confiance
    
    Args:
        nb_simul (int): Nombre de simulations
        barriere (bool): Option barrière ou non
        So (float): Prix initial de l'actif
        r (float): Taux sans risque annuel
        T (float): Maturité
        sigma (float): Volatilité annuel
        K (float): Prix d'exercice (strike)
        alpha (float): Niveau de signification pour l'IC
    
    Returns:
        tuple: (Prix de l'option, Intervalle de confiance)
    """
    x1 = (1/sigma*mt.sqrt(T))*(mt.log(K/So) - (r-(sigma**2/2))*T)
    prix_theorique = mt.exp(-r*T)*K*fonction_repart(x1) - So*fonction_repart(x1-sigma*mt.sqrt(T))

    if not barriere :
       
       # Simulation des trajectoires ( valeurs terminales )
       gaussien_w_T = simuler_W(nb_simul, T=2)
       #plot_graph(gaussien_w_T)
       
       #Calcul du prix St de l'option
       ST = So * np.exp( (r - (sigma**2)/2) * T + sigma *gaussien_w_T )

       # Calcul des payoffs
       Y = np.maximum(0, K - ST)
    
       # Prix actualisé de l'option
       P_euro = np.mean(Y) * np.exp(-r*T)

       # Calcul de la variance de Monte Carlo
       sigma_Y = (1 / (nb_simul - 1)) * sum((Y_j - np.mean(Y))**2 for Y_j in Y) 

       # Utilisation de la loi de Student 
       ddl = nb_simul - 1  # degrés de liberté
       t_value = stats.t.ppf(1 - alph/2, ddl)
        
       # Erreur standard
       std_error = np.sqrt(sigma_Y / nb_simul)
        
       # Intervalle de confiance
       IC_bas  = (np.mean(Y) - t_value * std_error)* np.exp(-r*T)
       IC_haut = (np.mean(Y) + t_value * std_error)* np.exp(-r*T)

       print(f"\nPrix de l'option PUT européenne : {P_euro:.4f} pour {nb_simul} trajectoires")
       print(f"Intervalle de confiance à {(1-alph)*100}%: [{IC_bas:.4f}, {IC_haut:.4f}]")
       print(f"Prix théorique de l'option PUT européenne: {prix_theorique:.4f}\n")
       

    return P_euro, (IC_bas, IC_haut), prix_theorique


def calcul_P_euro_trajectoires(nb_simul_list, So=1, r=0.015, T=2, sigma=0.15, K=1, alph=0.1):

    prices = []
    lower_bounds = []
    upper_bounds = []
    prix_theorique = 0.0
    
    for nb_simul in nb_simul_list:
        P_euro, (IC_bas, IC_haut), prix_theorique = calcul_P_euro(nb_simul, So=So, r=r, T=T, sigma=sigma, K=K, alph=alph)
        prices.append(P_euro)
        lower_bounds.append(IC_bas)
        upper_bounds.append(IC_haut)

    # Tracer les résultats
    plt.figure(figsize=(10, 6))
    plt.plot(nb_simul_list, prices, label="Prix estimé de l'option", color="blue", marker='o')

    # Tracer les intervalles de confiance
    plt.fill_between(nb_simul_list, lower_bounds, upper_bounds, color="blue", alpha=0.1, label="Intervalle de confiance 90%")

     # Ajouter une ligne horizontale pour le prix théorique
    plt.axhline(y=prix_theorique, color='red', linestyle='--', label=f"Prix théorique : {prix_theorique:.4f}")

    # Ajouter des détails au graphique
    plt.title("Estimation du prix d'une option européenne vanille avec intervalle de confiance")
    plt.xlabel("Nombre de simulations")
    plt.ylabel("Prix estimé de l'option")
    plt.legend()
    plt.grid(True)
    plt.show()
