import math as mt
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.linalg import cholesky





# ========================================
# Trace un graphique du prix estimé de l'option
# avec barres d'erreurs et IC (et prix théorique)
# 
# Entrées :
# - nb_simul_list : liste du nombre de simulations testées
# - barriere : booléen indiquant si l'option est à barrière
# - prices : liste des prix estimés par Monte Carlo
# - lower_bounds, upper_bounds : bornes des intervalles de confiance
# - prix_theorique : valeur théorique connue (0 si inconnu)
# 
# Sortie : aucun retour, mais affiche le graphique
# ========================================

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






# ====================================================
# Approximation numérique de la fonction de répartition
# de la loi normale centrée réduite (CDF de N(0,1)) pour x > 0
#
# Entrée : x (float)
# Sortie : valeur approchée de P(X ≤ x) pour X ~ N(0,1)
# ====================================================

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
    FDR = 1 - (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2) * poly

    return FDR






# ==================================================================
# Simulation de mouvements browniens simples ou via Cholesky
#
# Entrées :
# - nb_simul : nombre de trajectoires simulées
# - T : horizon temporel
# - barriere : si True, méthode de Cholesky
# - delta : pas de discrétisation
#
# Sortie :
# - tableau numpy des valeurs simulées de W_t
# ==================================================================

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
        N_delta = np.maximum( mt.floor(T/delta), 1)
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
    





# =============================================================
# Affiche un histogramme d'une simulation gaussienne (Box-Muller)
# avec courbe de densité gaussienne théorique en surimpression
#
# Entrée : resultats (array) des valeurs simulées
# Sortie : aucune, affiche le graphique
# =============================================================

def plot_graph_gaussienne(resultats):
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
    
    plt.title(f'Distribution des Nombres Gaussiens avec {len(resultats)} de tirages')
    plt.xlabel('Valeurs')
    plt.ylabel('Densité de Probabilité')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
   
    # Affichage de la figure
    plt.show()






# =========================================================
# Trace plusieurs trajectoires browniennes simulées
#
# Entrées :
# - nb_simul : nombre de trajectoires
# - T : maturité
# - delta : pas de temps
# - barriere : méthode de simulation à barrière (Cholesky)
#
# Sortie : aucune, affiche les trajectoires simulées
# =========================================================

def plot_trajectoires_browniennes(nb_simul=10, T=2, delta=1/52, barriere = True):
    """
    Trace plusieurs trajectoires browniennes simulées avec des couleurs différentes.
    """
    N_delta = np.maximum( mt.floor(T/delta), 1)
    time_grid = np.linspace(delta, T, N_delta)
    
    trajectoires = simuler_W(nb_simul, T=T, barriere=barriere, delta=delta)

    W0 = np.zeros((nb_simul, 1))
    trajectoires = np.concatenate((W0, trajectoires), axis=1)
    time_grid = np.concatenate(([0], time_grid))

    
    
    plt.figure(figsize=(10, 6))
    
    for i in range(nb_simul):
        plt.plot(time_grid, trajectoires[i], label=f"Traj {i+1}")
    
    plt.title("Trajectoires simulées du mouvement brownien")
    plt.xlabel("Temps")
    plt.ylabel("Valeur du mouvement brownien")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()






# ========================================================================
# Simulation et tracé des trajectoires S(t) du sous-jacent
# selon le modèle de Black-Scholes
#
# Entrées :
# - nb_simul : nombre de trajectoires
# - S0 : prix initial
# - K : strike
# - r : taux sans risque
# - sigma : volatilité
# - T : maturité
# - delta : pas de temps
# - B : niveau de barrière
# - m_covariance : True => Cholesky sinon somme gaussienne
#
# Sortie :
# - pourcentage de trajectoires In The Money (ST < K et min S(u) ≥ B)
# - graphique des trajectoires avec lignes K, S0 et B
# ========================================================================

def tracer_trajectoires_st(nb_simul=50, S0=1, K=1, r=0.015, sigma=0.15, T=2, delta=1/52, B= 0.7, m_covariance=True):
    N_delta = max(mt.floor(T/delta), 1)
    time_grid = np.linspace(delta, T, N_delta)
    
    # Simuler les trajectoires de W_t
    if m_covariance:
        T_list = [(i+1)*delta for i in range(N_delta)]
        G = np.random.normal(0, 1, (N_delta, nb_simul))
        Gamma = np.zeros((N_delta, N_delta))
        I = np.zeros((N_delta, N_delta))
        for i in range(N_delta):
            Gamma[i, i:N_delta] = T_list[i]
            I[i, i] = T_list[i]
        Gamma = Gamma + Gamma.T - I
        A = cholesky(Gamma)
        W = np.dot(A, G).T
    else:
        W = np.cumsum(np.random.normal(0, np.sqrt(delta), (nb_simul, N_delta)), axis=1)
    
    # Calcul de S_t
    T_path = np.broadcast_to(np.array([(i+1)*delta for i in range(N_delta)]), (nb_simul, N_delta))
    S = S0 * np.exp((r - 0.5 * sigma**2) * T_path + sigma * W)

    # Ajouter S0 à t=0
    S = np.concatenate((S0 * np.ones((nb_simul, 1)), S), axis=1)
    time_grid = np.concatenate(([0], time_grid))


    # Pourcentage des trajectoires avec S(T) < K et min S(u)>=B
    S_min  = np.min(S, axis = 1)
    ST = S[:, -1]
    pourcentage_ITM = 100 * np.sum(ST < K)* np.sum(S_min >= B) / (nb_simul)**2

    # Tracer les trajectoires
    plt.figure(figsize=(10, 6))
    for i in range(nb_simul):
        plt.plot(time_grid, S[i])
    
    plt.axhline(y=K, color='red', linestyle='--', label=f"Strike K = {K}")
    plt.axhline(y=S0, color='green', linestyle='--', label=f"S0 = {S0}")
    plt.axhline(y=B, color='blue', linestyle='--', label=f"B = {B}")
    
    plt.title(f"Trajectoires simulées de S(t) valides ({pourcentage_ITM:.1f}% ITM) pour {nb_simul} trajectoires")
    plt.xlabel("Temps")
    plt.ylabel("Cours de l'actif S(t)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return pourcentage_ITM



