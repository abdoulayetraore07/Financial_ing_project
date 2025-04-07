import math as mt
import random
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from numpy.linalg import cholesky



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



def calcul_P_euro(nb_simul, barriere=False, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph = 0.1, delta = 1/52, B = 0.7, m_covariance = True):
    
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
   

    if not barriere :
        
       # Simulation des trajectoires ( valeurs terminales T )
       G = simuler_W(nb_simul, T=2)
       
       #Calcul du prix St de l'option
       S = S0 * np.exp( (r - (sigma**2)/2) * T + sigma *np.sqrt(T)*G)

       # Calcul des payoffs
       payoff = np.exp(-r*T)*np.maximum(0, K - S) 

    else :
       
        # Calculs des parametres de discretisation
        N_delta = mt.floor(T/delta)
      
        if not m_covariance :
            
            #Calcul des réalistions de St pour N_delta instances du brownien et nb_simul simulations 
            G = np.random.normal(0,1,(nb_simul, N_delta))
            LR = (r-sigma**2/2)*delta + sigma*np.sqrt(delta)*G
            #LR = np.concatenate((np.ones((nb_simul,1))*np.log(S0), LR), axis = 1) On enlève car dans la formule on tient compte que des temps T1, ,TN_delta : pas de S0
                
            log_path = np.cumsum(LR, axis=1)
            
            #Obtention des trajectoires 
            S_path  = np.exp(log_path)
        
        else :
            #Calcul des réalistions de W_t pour N_delta instances du brownien et nb_simul simulations 
            gaussien_W_path = simuler_W(nb_simul, T=2, barriere = True, delta = delta)

            #Obtention des trajectoires 
            S_path  = S0 * np.exp( (r - (sigma**2)/2) * T + sigma*gaussien_W_path )
        
        #Obtention du vecteur des mins sur chaque trajectoire 
        S_path_min  = np.min(S_path, axis = 1)
          
        # Calcul des payoffs
        payoff = np.exp(-r*T)*np.maximum(0, K - S_path[:,-1])*(S_path_min>=B)


    #ESTIMATION GLOBLALE DU PRIX DE L'OPTION PAR MONTE-CARLO

    # Prix estimé de l'option via MC
    MC_price = np.mean(payoff)

    # Calcul de la variance de Monte Carlo
    sigma_payoff = (1 / (nb_simul - 1)) * sum((Y_j - MC_price)**2 for Y_j in payoff) 

    # Utilisation de la loi de Student 
    ddl = nb_simul - 1  # degrés de liberté
    t_value = stats.t.ppf(1 - alph/2, ddl)
    
    # Erreur standard
    std_error = np.sqrt(sigma_payoff/ nb_simul)
    
    # Intervalle de confiance
    IC_bas  = MC_price - t_value * std_error
    IC_haut = MC_price + t_value * std_error

    print(f"\nPrix de l'option PUT européenne par MC: {MC_price:.4f} pour {nb_simul} trajectoires")
    print(f"Intervalle de confiance à {(1-alph)*100}%: [{IC_bas:.4f}, {IC_haut:.4f}]")
    print(f"L'erreur standard de l'estimation est : {std_error}\n")

    return MC_price, (IC_bas, IC_haut)


def calcul_P_euro_trajectoires(nb_simul_list, barriere=False, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph=0.1, delta = 1/52 ,B = 0.7, m_covariance = True ):

    prices = []
    lower_bounds = []
    upper_bounds = []

    for nb_simul in nb_simul_list:
        MC_price, (IC_bas, IC_haut) = calcul_P_euro(nb_simul, barriere= barriere, S0=S0, r=r, T=T, sigma=sigma, K=K, alph = alph, delta = delta, B = B, m_covariance = m_covariance)
        prices.append( MC_price)
        lower_bounds.append(IC_bas)
        upper_bounds.append(IC_haut)


    if not barriere:
        x1 = (1/sigma*np.sqrt(T))*(mt.log(K/S0) - (r-(sigma**2/2))*T)
        prix_theorique = np.exp(-r*T)*K*fonction_repart(x1) - S0*fonction_repart(x1-sigma*mt.sqrt(T))
    else :
        prix_theorique = 0

    
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


