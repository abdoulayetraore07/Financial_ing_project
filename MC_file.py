import math as mt
import numpy as np
import random
import scipy.stats as stats
from fonctions import *






# ========================================
# Simule les trajectoires du cours S(t) selon le modèle de Black-Scholes
# avec possibilité de méthode par covariance (pont brownien)
# et de réduction de variance (antithétique)
#
# Entrées :
# - nb_simul : nombre de simulations
# - S0 : prix initial
# - r : taux sans risque
# - T : maturité de l'option
# - sigma : volatilité de l'actif
# - delta : pas de discrétisation
# - m_covariance : booléen pour méthode par covariance (True = pont brownien)
# - r_variance : booléen pour activer la réduction de variance
#
# Sortie :
# - S_path : trajectoires standards
# - S_path_neg : trajectoires inverses si r_variance est activé, sinon identique à S_path
# ========================================

def get_S_path(nb_simul, S0=1, r=0.015, T=2, sigma=0.15, delta = 1/52, m_covariance = False, r_variance = False):
    # Calculs des parametres de discretisation
    N_delta = np.maximum(mt.floor(T/delta), 1)
    S_path_neg  = np.zeros((nb_simul, N_delta))

    if not m_covariance : 
        #Calcul des réalistions de St pour N_delta instances du brownien et nb_simul simulations 
        G = np.random.normal(0,1,(nb_simul, N_delta))
        LR = (r-sigma**2/2)*delta + sigma*np.sqrt(delta)*G   
        log_path = np.cumsum(LR, axis=1) 
        S_path  = np.exp(log_path)

        if r_variance:
            LR_neg = (r-sigma**2/2)*delta + sigma*np.sqrt(delta)*(-G)
            log_path_neg = np.cumsum(LR_neg, axis=1)
            S_path_neg  = np.exp(log_path_neg)

    else :
        #Calcul des réalistions de W_t pour N_delta instances du brownien et nb_simul simulations 
        gaussien_W_path = simuler_W(nb_simul, T=2, barriere = True, delta = delta)

        #Obtention des trajectoires 
        T_path = [(i+1)*delta for i in range(N_delta)]             
        T_path =  np.broadcast_to(T_path , (nb_simul, N_delta))    ## Fonction pour repeter le meme vecteur sur chaque ligne (On get une matrice)
        S_path  = S0 * np.exp( (r - (sigma**2)/2) * T_path + sigma*gaussien_W_path )

        if r_variance:
            S_path_neg  = S0 * np.exp( (r - (sigma**2)/2) * T_path + sigma*(-gaussien_W_path) )

    #Conditionnement pour avoir une formule plus compacte 
    S_path_neg =  S_path_neg*(r_variance==True) + S_path*(r_variance==False)

    return S_path, S_path_neg






# ========================================
# Calcule le prix d'une option et son intervalle de confiance
# via la méthode de Monte Carlo (avec Student)
#
# Entrées :
# - payoff : vecteur des gains simulés
# - nb_simul : nombre de simulations
# - alph : niveau de risque pour l'intervalle de confiance
# - calcul_proba : booléen, si True n'affiche pas les impressions
#
# Sortie :
# - MC_price : prix moyen estimé
# - (IC_bas, IC_haut) : bornes de l'intervalle de confiance
# ========================================

def compute_MC(payoff, nb_simul,  alph, calcul_proba = False):
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

    if not calcul_proba :
        print(f"\nPrix de l'option PUT européenne par MC: {MC_price:.4f} pour {nb_simul} trajectoires")
        print(f"Intervalle de confiance à {(1-alph)*100}%: [{IC_bas:.4f}, {IC_haut:.4f}]")
        print(f"L'erreur standard de l'estimation est : {std_error}\n")

    return MC_price, (IC_bas, IC_haut)






# ========================================
# Évalue les prix d'une option pour différentes tailles d’échantillons
# et affiche les résultats avec intervalles de confiance
#
# Entrées :
# - nb_simul_list : liste de tailles de simulation
# - barriere : booléen pour option barrière
# - S0, r, T, sigma, K : paramètres Black-Scholes
# - alph : niveau de risque pour l'IC
# - delta : pas de discrétisation
# - B : niveau de la barrière
# - m_covariance : méthode pont brownien
# - r_variance : réduction de variance
# - comparer_r_var : active la comparaison entre réduction ou non
#
# Sortie :
# - Aucun retour, affiche les graphiques
# ========================================

def calcul_P_euro_et_DO_delta(nb_simul, barriere=False, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph = 0.1, delta = 1/52, B = 0.7, m_covariance = False, r_variance = False):
    
    """
    Calcul du prix d'une option européenne PUT avec intervalle de confiance
    
    Args:
        nb_simul (int): Nombre de simulations
        barriere (bool): Option barrière ou non
        So (float): Prix initial de l'actif
        r (float): Taux sans risque annuel
        T (float): Maturité annuel
        sigma (float): Volatilité annuel
        K (float): Prix d'exercice (strike)
        alpha (float): Niveau de signification pour l'IC
    
    Returns:
        tuple: (Prix de l'option, Intervalle de confiance)
    """
   

    if not barriere :
        
       # Simulation des trajectoires ( valeurs terminales T )
       G = simuler_W(nb_simul, T=2)
       
       #Calcul du prix ST de l'option
       ST = S0 * np.exp( (r - (sigma**2)/2) * T + sigma *np.sqrt(T)*G)

       # Calcul des payoffs
       payoff = np.exp(-r*T)*np.maximum(0, K - ST) 

    else :
        
        #Obtention des trajectoires
        S_path, S_path_neg = get_S_path(nb_simul, S0=S0, r=r, T=T, sigma=sigma, delta = delta, m_covariance = m_covariance, r_variance = r_variance)

        #Obtention du vecteur des mins sur chaque trajectoire 
        S_path_min  = np.min(S_path, axis = 1)
        S_path_neg_min  = np.min(S_path_neg, axis = 1)

        # Calcul des payoffs
        payoff = 0.5* np.exp(-r*T) *(  np.maximum(0, K - S_path[:,-1])*(S_path_min>=B)   +   np.maximum(0, K - S_path_neg[:,-1])*(S_path_neg_min>=B  ) )


    #ESTIMATION GLOBLALE DU PRIX DE L'OPTION PAR MONTE-CARLO
    MC_price, (IC_bas, IC_haut) = compute_MC(payoff, nb_simul,  alph)

    return MC_price, (IC_bas, IC_haut)






# ========================================
# Évalue les prix d'une option pour différentes tailles d’échantillons
# et affiche les résultats avec intervalles de confiance
#
# Entrées :
# - nb_simul_list : liste de tailles de simulation
# - barriere : booléen pour option barrière
# - S0, r, T, sigma, K : paramètres Black-Scholes
# - alph : niveau de risque pour l'IC
# - delta : pas de discrétisation
# - B : niveau de la barrière
# - m_covariance : méthode pont brownien
# - r_variance : réduction de variance
# - comparer_r_var : active la comparaison entre réduction ou non
#
# Sortie :
# - Aucun retour, affiche les graphiques
# ========================================

def calcul_P_trajectoires(nb_simul_list, barriere=False, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph=0.1, delta = 1/52 ,B = 0.7, m_covariance = False, r_variance = False, comparer_r_var = False):

    prices = []
    lower_bounds = []
    upper_bounds = []

    for nb_simul in nb_simul_list:
        MC_price, (IC_bas, IC_haut) = calcul_P_euro_et_DO_delta(nb_simul, barriere= barriere, S0=S0, r=r, T=T, sigma=sigma, K=K, alph = alph, delta = delta, B = B, m_covariance = m_covariance, r_variance= r_variance)
        prices.append( MC_price)
        lower_bounds.append(IC_bas)
        upper_bounds.append(IC_haut)
    
    if not barriere:
        x1 = (1/sigma*np.sqrt(T))*(np.log(K/S0) - (r-(sigma**2/2))*T)
        prix_theorique = np.exp(-r*T)*K*fonction_repart(x1) - S0*fonction_repart(x1-sigma*mt.sqrt(T))
    else :
        prix_theorique = 0

    if r_variance == True  and comparer_r_var==True :
        prices_2 = []
        lower_bounds_2 = []
        upper_bounds_2 = []

        for nb_simul in nb_simul_list:
            MC_price, (IC_bas, IC_haut) = calcul_P_euro_et_DO_delta(nb_simul, barriere= barriere, S0=S0, r=r, T=T, sigma=sigma, K=K, alph = alph, delta = delta, B = B, m_covariance = m_covariance, r_variance= False)
            prices_2.append( MC_price)
            lower_bounds_2.append(IC_bas)
            upper_bounds_2.append(IC_haut)
        
        comparer_r_variance(nb_simul_list, prices, lower_bounds, upper_bounds, prices_2, lower_bounds_2, upper_bounds_2)
        
    else :    
        plot_graph_trajet(nb_simul_list, barriere,  prices, lower_bounds, upper_bounds, prix_theorique)
    

    



# ========================================
# Compare les résultats avec et sans réduction de variance
# sur plusieurs tailles d’échantillons (Monte Carlo)
#
# Entrées :
# - nb_simul_list : liste des tailles
# - prices_1, prices_2 : prix estimés avec et sans VC
# - lower_bounds, upper_bounds : intervalles de confiance pour chaque méthode
#
# Sortie :
# - Aucun retour, affiche les courbes comparatives
# ========================================

def comparer_r_variance(nb_simul_list, prices_1, lower_bounds_1, upper_bounds_1, prices_2, lower_bounds_2, upper_bounds_2):
    # Tracer les résultats
    plt.figure(figsize=(10, 6))
    plt.plot(nb_simul_list, prices_1, label="Prix estimé de l'option avec reduction de variance", color="blue", marker='o')
    plt.plot(nb_simul_list, prices_2, label="Prix estimé de l'option sans reduction de variance", color="green", marker='o')

    # Tracer les intervalles de confiance
    plt.fill_between(nb_simul_list, lower_bounds_1, upper_bounds_1, 
                 color="blue", alpha=0.2, 
                 edgecolor="black", linewidth=1.5,
                 label="Intervalle de confiance 90% (méthode 1)")

    plt.fill_between(nb_simul_list, lower_bounds_2, upper_bounds_2, 
                 color="green", alpha=0.2, 
                 edgecolor="black", linewidth=1.5,
                 label="Intervalle de confiance 90% (méthode 2)")
    # Ajouter une ligne horizontale pour le prix théorique
    plt.title("Comparaison des estimations du prix d'une option barriere européenne vanille avec et sans reduction de variance")
    plt.xlabel("Nombre de simulations")
    plt.ylabel("Prix estimé de l'option") 

    plt.legend()
    plt.grid(True)
    plt.show()






# ========================================
# Évalue le prix d'une option barrière
# en fonction de différentes valeurs de barrière B
#
# Entrées :
# - valeurs_B : liste de seuils B à tester
# - nb_simul : nombre de simulations
#
# Sortie :
# - Aucun retour, affiche un graphique du prix en fonction de B
# ========================================

def comparer_B(valeurs_B, nb_simul = 120000):
    prices = []
    for B in valeurs_B:
        MC_price, (IC_bas, IC_haut) = calcul_P_euro_et_DO_delta(nb_simul, barriere=True, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph = 0.1, delta = 1/52, B = B, m_covariance = False, r_variance = True)
        prices.append(MC_price)
    
    plt.figure(figsize=(10, 6))
    plt.plot(valeurs_B, prices, label="Prix estimé de l'option en fonction de B", color="blue", marker='o')

    plt.title("Estimations du prix d'une option barriere européenne vanille en fonction de B avec reduction de variance")
    plt.xlabel("valeurs de B")
    plt.ylabel("Prix estimé de l'option") 

    plt.legend()
    plt.grid(True)
    plt.show()






# ========================================
# Calcule le prix de l'option barrière pour différentes valeurs de sigma
# avec comparaison possible entre deux valeurs de S0 (1 et 0.8)
#
# Entrées :
# - valeurs_sigma : liste de volatilités testées
# - nb_simul : nombre de simulations
# - S0 : prix initial
# - comparer : booléen pour comparer S0=1 vs S0=0.8
#
# Sortie :
# - Aucun retour, affiche les courbes
# ========================================

def comparer_sigma(valeurs_sigma, nb_simul = 120000, S0 = 1, comparer = False):
    prices = []
    prices_2 = []

    if not comparer :
        for sigma in valeurs_sigma:
            
            MC_price , (IC_bas, IC_haut) = calcul_P_euro_et_DO_delta(nb_simul, barriere=True, S0=S0, r=0.015, T=2, sigma=sigma, K=1, alph = 0.1, delta = 1/52, B = 0.7, m_covariance = False, r_variance = True)
            prices.append(MC_price)
    else :
        for sigma in valeurs_sigma:

            MC_price , (IC_bas, IC_haut) = calcul_P_euro_et_DO_delta(nb_simul, barriere=True, S0=1, r=0.015, T=2, sigma=sigma, K=1, alph = 0.1, delta = 1/52, B = 0.7, m_covariance = False, r_variance = True)
            prices.append(MC_price)
            MC_price , (IC_bas, IC_haut) = calcul_P_euro_et_DO_delta(nb_simul, barriere=True, S0=0.8, r=0.015, T=2, sigma=sigma, K=1, alph = 0.1, delta = 1/52, B = 0.7, m_covariance = False, r_variance = True)
            prices_2.append(MC_price)
        

    plt.figure(figsize=(10, 6))


    if comparer :
        plt.plot(valeurs_sigma, prices, label="Prix estimé de l'option en fonction de sigma avec S0 = 1", color="blue", marker='o')
        plt.plot(valeurs_sigma, prices_2, label="Prix estimé de l'option en fonction de sigma avec S0 = 0.8", color="green", marker='o')
    else :
        plt.plot(valeurs_sigma, prices, label=f"Prix estimé de l'option en fonction de sigma avec S0 = {S0}", color="blue", marker='o')
        plt.plot([], [], ' ', label=f"S0 = {S0}", marker='o')


    plt.title("Estimations du prix d'une option barriere européenne vanille en fonction de sigma avec reduction de variance")
    plt.xlabel("Valeurs de sigma")
    plt.ylabel("Prix estimé de l'option") 


    plt.legend()
    plt.grid(True)
    plt.show()






# ========================================
# Corrige la discrétisation d'une barrière en calculant
# la probabilité de franchissement via un pont brownien
#
# Entrées :
# - S_path : matrice des trajectoires simulées
# - B : niveau de la barrière
# - sigma : volatilité de l'actif
# - delta : pas de discrétisation en temps
# - r : taux d’intérêt sans risque
#
# Sortie : vecteur contenant les probabilités de non-franchissement
# ========================================

def correction_discretisation(S_path, B=0.7, sigma=0.15, delta=1/52 , r= 0.015):
    
    nb_simul, N_delta= S_path.shape[0], S_path.shape[1]
    barriere_atteinte = np.zeros(nb_simul, dtype=int)
    
    for i in range(nb_simul):
        proba_total_traject = 1
            
        for j in range(N_delta-1):
            # Si déjà touché la barrière ou si les prix sont inférieurs à la barrière
            if S_path[i, j] <= B or S_path[i, j+1] <= B:
                proba_total_traject = 0
                break
            
            # Cas ou on descends pas en dessous de B pour les extrémités
            h = np.log( S_path[i, j]/B  + delta*(r-sigma**2/2) ) *  np.log(S_path[i, j+1]/B)  #

            # Probabilité de franchissement avec le pont brownien
            proba_total_traject *= (1 - np.exp(-2 * h / (sigma**2 * delta)))
            
        barriere_atteinte[i] = proba_total_traject

    return barriere_atteinte  



"""
def correction_discretisation(S_path, B=0.7, sigma=0.15, delta=1/52):
    nb_simul, N_delta= S_path.shape[0], S_path.shape[1]
    barriere_atteinte = np.zeros(nb_simul, dtype=bool)
    
    for i in range(nb_simul):
        for j in range(N_delta-1):
            # Si déjà touché la barrière ou si les prix sont inférieurs à la barrière
            if barriere_atteinte[i] or S_path[i, j] <= B or S_path[i, j+1] <= B:
                barriere_atteinte[i] = True
                break
            
            # Cas ou on descends pas en dessous de B pour les extrémités
            h = np.log( S_path[i, j]/B) * np.log( S_path[i, j+1]/B)

            # Probabilité de franchissement avec le pont brownien
            p_barriere = np.exp(-2 * h / (sigma**2 * delta))
            if random.uniform(0.0, 1.0) < p_barriere:
                barriere_atteinte[i] = True
                break

    barriere_non_atteinte = ~barriere_atteinte # True pour les chemins qui n'ont pas touché la barrière, juste pratique pour la suite

    return barriere_non_atteinte  

    

def correction_discretisation(S_path, B=0.7, sigma=0.15, delta=1/52 , r= 0.015):
    
    nb_simul, N_delta= S_path.shape[0], S_path.shape[1]
    barriere_atteinte = np.zeros(nb_simul, dtype=int)
    
    for i in range(nb_simul):
        proba_total_traject = 1
            
        for j in range(N_delta-1):
            # Si déjà touché la barrière ou si les prix sont inférieurs à la barrière
            if S_path[i, j] <= B or S_path[i, j+1] <= B:
                proba_total_traject = 0
                break
            
            # Cas ou on descends pas en dessous de B pour les extrémités
            h = np.log( S_path[i, j]/B ) *  np.log(S_path[i, j+1]/B)  #

            # Probabilité de franchissement avec le pont brownien
            proba_total_traject *= (1 - np.exp(-2 * h / (sigma**2 * delta)))
            
        barriere_atteinte[i] = proba_total_traject

    return barriere_atteinte  
"""






# ========================================
# Calcule le prix d'une option barrière (Down & Out)
# en utilisant les trajectoires corrigées (pont brownien)
#
# Entrées :
# - nb_simul : nombre de simulations
# - S0, r, T, sigma, K, alph : paramètres standards de l’option
# - delta, B : discrétisation et niveau de barrière
# - m_covariance, r_variance : booléens pour méthode de covariance et réduction de variance
#
# Sortie : tuple (prix estimé, intervalle de confiance)
# ========================================

def calcul_P_DO(nb_simul=120000, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph = 0.1, delta = 1/52, B = 0.7, m_covariance = False, r_variance = True):

    #Obtention des trajectoires
    S_path, S_path_neg = get_S_path(nb_simul, S0=S0, r=r, T=T, sigma=sigma, delta = delta, m_covariance = m_covariance, r_variance = r_variance)

    #Obtention des vecteurs disant pour chaque trajectoire si on a touché la barriere ou pas  
    S_path_hit  = correction_discretisation(S_path=S_path, B=B, sigma=sigma, delta=delta)
    S_path_neg_hit  = correction_discretisation(S_path=S_path_neg, B=B, sigma=sigma, delta=delta)

    # Calcul des payoffs
    payoff = 0.5 * np.exp(-r*T) * (  np.maximum(0, K - S_path[:,-1])*S_path_hit   +   np.maximum(0, K - S_path_neg[:,-1])*S_path_neg_hit )

    #ESTIMATION GLOBLALE DU PRIX DE L'OPTION PAR MONTE-CARLO
    MC_price, (IC_bas, IC_haut) = compute_MC(payoff, nb_simul,  alph)

    return MC_price, (IC_bas, IC_haut)






# ========================================
# Trace le prix estimé de l'option P_DO
# en fonction du nombre de trajectoires
#
# Entrées :
# - nb_simul_list : liste des tailles d’échantillons testées
# - S0, r, T, sigma, K, alph : paramètres standards de l’option
# - delta, B : discrétisation et niveau de barrière
# - m_covariance, r_variance : booléens pour méthode de covariance et réduction de variance
#
# Sortie : graphique des prix estimés
# ========================================

def calcul_P_DO_trajectoires(nb_simul_list, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph=0.1, delta = 1/52 ,B = 0.7, m_covariance = False, r_variance = True):
    prices = []
    lower_bounds = []
    upper_bounds = []

    for nb_simul in nb_simul_list:
        MC_price, (IC_bas, IC_haut) = calcul_P_DO(nb_simul, S0=S0, r=r, T=T, sigma=sigma, K=K, alph = alph, delta = delta, B = B, m_covariance = m_covariance, r_variance= r_variance)
        prices.append( MC_price)
        lower_bounds.append(IC_bas)
        upper_bounds.append(IC_haut)

    
    plot_graph_trajet(nb_simul_list, True,  prices, lower_bounds, upper_bounds)






# ========================================
# Compare les prix obtenus via la méthode P_DO
# et la méthode P_DO_delta (discrétisation simple)
# pour différentes valeurs de delta
#
# Entrées :
# - valeurs_delta : liste de deltas à tester
# - delta_DO : valeur de référence pour la méthode corrigée
# - autres paramètres standards de l’option
#
# Sortie : graphique comparatif
# ========================================

def comparer_DO_et_DO_delta(valeurs_delta = [1/52], nb_simul = 120000, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph=0.1, B = 0.7, m_covariance = False, r_variance = True,  delta_DO = 1/250):
 
    prices_DO_delta = []
    #prices_DO = []

    MC_price_DO, (IC_bas, IC_haut) = calcul_P_DO(nb_simul=nb_simul, S0=S0, r=r, T=T, sigma=sigma, K=K, alph = alph, delta = delta_DO, B = B, m_covariance = m_covariance, r_variance = r_variance)
      
    for delta in valeurs_delta:
        MC_price_DO_Delta, (IC_bas, IC_haut) = calcul_P_euro_et_DO_delta(nb_simul=nb_simul, barriere= True, S0=S0, r=r, T=T, sigma=sigma, K=K, alph = alph, delta = delta, B = B, m_covariance = m_covariance, r_variance= r_variance)
        prices_DO_delta.append(MC_price_DO_Delta)
        """
        MC_price_DO, (IC_bas, IC_haut) = calcul_P_DO(nb_simul=nb_simul, S0=S0, r=r, T=T, sigma=sigma, K=K, alph = alph, delta = delta, B = B, m_covariance = m_covariance, r_variance = r_variance)
        prices_DO.append(MC_price_DO)
        """
           
    plt.figure(figsize=(10, 6))

    plt.plot(valeurs_delta, prices_DO_delta, label="Prix estimé de l'option avec P_DO_delta en fonction de delta ", color="blue", marker='o')
    plt.axhline(y=MC_price_DO, color='green', linestyle='--', label=f"label=Prix estimé de l'option avec P_DO pour delta = {delta_DO}")
    #plt.plot(valeurs_delta,prices_DO, label="Prix estimé de l'option avec P_DO en fonction de delta", color="green", marker='o')
    
    plt.title("Estimations du prix d'une option barriere européenne vanille en fonction de delta avec reduction de variance")
    plt.xlabel("Valeurs de delta")
    plt.ylabel("Prix estimé de l'option") 


    plt.legend()
    plt.grid(True)
    plt.show()






# ========================================
# Compare les probabilités de non-franchissement
# entre la méthode P_DO (pont brownien) et P_DO_delta
#
# Entrées :
# - valeurs_delta : liste des deltas à tester
# - delta_DO : valeur utilisée pour le pont brownien
# - autres paramètres standards de l’option
#
# Sortie : graphique des probabilités
# ========================================

def comparer_proba_non_sortie_delta(valeurs_delta = [1/52], nb_simul = 120000, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph = 0.1, B = 0.7, m_covariance = False, r_variance = False, delta_DO = 1/250):

    Proba_in_DO_delta = []
    cpt = 1 

    
    for delta in  valeurs_delta :
        #Obtention des trajectoires
        S_path, S_path_neg = get_S_path(nb_simul, S0=S0, r=r, T=T, sigma=sigma, delta = delta, m_covariance = m_covariance, r_variance = r_variance)

        #Obtention du vecteur des mins sur chaque trajectoire
        S_path_min  = np.min(S_path, axis = 1)
        
        if delta == delta_DO :
            #Obtention du vecteur pont gaussien sur chaque trajectoire
            S_path_hit  = correction_discretisation(S_path=S_path, B=B, sigma=sigma, delta=delta)
            payoff_DO = S_path_hit   
            proba_in_DO, (IC_bas, IC_haut) = compute_MC(payoff=payoff_DO, nb_simul=nb_simul,  alph=alph, calcul_proba= True)
    
        # Calcul des payoffs
        payoff_DO_delta = (S_path_min>=B) 
       
        #ESTIMATION GLOBLALE DU PRIX DE L'OPTION PAR MONTE-CARLO
        proba_in_DO_delta, (IC_bas, IC_haut) = compute_MC(payoff=payoff_DO_delta, nb_simul=nb_simul,  alph=alph, calcul_proba= True)
        
        Proba_in_DO_delta.append(proba_in_DO_delta)
        print(f"Etapes : {cpt} / {len(valeurs_delta)} terminées")
        cpt+=1

    print("\n")
    plt.figure(figsize=(10, 6))

    plt.plot(valeurs_delta,  Proba_in_DO_delta, label="Probabilité de non-sortie de ksi_DO_delta en fonction de delta", color="blue", marker='o')
    plt.axhline(y=proba_in_DO, color='green', linestyle='--', label=f"label=Prix estimé de l'option avec P_DO pour delta = {delta_DO}")
    
    plt.title("Probabilité de non-franchissement de barriere en fonction de delta")
    plt.xlabel("Valeurs de delta")
    plt.ylabel("Probabilité de non-franchissement") 


    plt.legend()
    plt.grid(True)
    plt.show()
    





# ========================================
# Applique la méthode de variable de contrôle
# pour estimer le prix d’une option barrière
#
# Entrées :
# - nb_simul : nombre de simulations
# - S0, r, T, sigma, K, alph : paramètres standards de l’option
# - delta, B : discrétisation et niveau de barrière
# - m_covariance, r_variance : booléens pour méthode de covariance et réduction de variance
#
# Sortie : tuple (prix estimé avec VC, intervalle de confiance)
# ========================================

def variable_controle( nb_simul=120000, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph = 0.1, delta = 1/52, B = 0.7, m_covariance = False, r_variance = True):

    #Obtention des trajectoires
    S_path, S_path_neg = get_S_path(nb_simul, S0=S0, r=r, T=T, sigma=sigma, delta = delta, m_covariance = m_covariance, r_variance = r_variance)

    #Obtention des vecteurs disant pour chaque trajectoire si on a touché la barriere ou pas  
    S_path_hit  = correction_discretisation(S_path=S_path, B=B, sigma=sigma, delta=delta)
    S_path_neg_hit  = correction_discretisation(S_path=S_path_neg, B=B, sigma=sigma, delta=delta)
    
    #Conversion en proba relatif à P_DI
    S_path_hit  = 1 - correction_discretisation(S_path=S_path, B=B, sigma=sigma, delta=delta)
    S_path_neg_hit  = 1 - correction_discretisation(S_path=S_path_neg, B=B, sigma=sigma, delta=delta)

    # Calcul des payoffs
    payoff = 0.5 * np.exp(-r*T) * (  np.maximum(0, K - S_path[:,-1])*S_path_hit   +   np.maximum(0, K - S_path_neg[:,-1])*S_path_neg_hit )

    #ESTIMATION GLOBLALE DU PRIX DE L'OPTION PAR MONTE-CARLO
    MC_price, (IC_bas, IC_haut) = compute_MC(payoff, nb_simul,  alph, calcul_proba = True)

    #Calcul de P_DI
    P_DI = MC_price

    #Calcul de E[Y] par méthode explicite
    x1 = (1/sigma*np.sqrt(T))*(np.log(K/S0) - (r-(sigma**2/2))*T)
    prix_theorique = np.exp(-r*T)*K*fonction_repart(x1) - S0*fonction_repart(x1-sigma*mt.sqrt(T))
    E_Y = prix_theorique

    #Calcul de P_DO
    P_DO = E_Y - P_DI
    IC_bas, IC_haut = E_Y - IC_haut, E_Y - IC_bas

    print(f"\nPrix de l'option PUT européenne par MC: {P_DO:.4f} pour {nb_simul} trajectoires")
    print(f"Intervalle de confiance à {(1-alph)*100}%: [{IC_bas:.4f}, {IC_haut:.4f}]")

    return P_DO, (IC_bas, IC_haut)






# ========================================
# Trace le prix estimé par méthode VC
# en fonction du nombre de simulations
#
# Entrées :
# - nb_simul_list : tailles d’échantillons testées
# - autres paramètres standards
#
# Sortie : graphique des estimations VC
# ========================================

def calcul_P_DO_VC_trajectoires(nb_simul_list, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph=0.1, delta = 1/52 ,B = 0.7, m_covariance = False, r_variance = True):
    prices = []
    lower_bounds = []
    upper_bounds = []

    for nb_simul in nb_simul_list:
        MC_price, (IC_bas, IC_haut) = variable_controle( nb_simul=nb_simul, S0= S0, r=r, T=T, sigma=sigma, K=K, alph = alph, delta = delta, B = B, m_covariance = m_covariance, r_variance = r_variance)
        prices.append( MC_price)
        lower_bounds.append(IC_bas)
        upper_bounds.append(IC_haut)

    
    plot_graph_trajet(nb_simul_list, True,  prices, lower_bounds, upper_bounds)






# ========================================
# Compare les prix estimés avec et sans
# variable de contrôle (VC)
#
# Entrées :
# - nb_simul_list : tailles d’échantillons testées
# - autres paramètres standards
#
# Sortie : graphique de comparaison
# ========================================

def comparer_VC_et_sans_VC(nb_simul_list, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph=0.1, delta = 1/52 ,B = 0.7, m_covariance = False, r_variance = True):

    prices_1 = []
    lower_bounds_1 = []
    upper_bounds_1 = []

    prices_2 = []
    lower_bounds_2 = []
    upper_bounds_2 = []

    for nb_simul in nb_simul_list:

        MC_price, (IC_bas, IC_haut) = calcul_P_DO(nb_simul, S0=S0, r=r, T=T, sigma=sigma, K=K, alph = alph, delta = delta, B = B, m_covariance = m_covariance, r_variance= r_variance)
        prices_1.append( MC_price)
        lower_bounds_1.append(IC_bas)
        upper_bounds_1.append(IC_haut)

        MC_price, (IC_bas, IC_haut) = variable_controle( nb_simul=nb_simul, S0= S0, r=r, T=T, sigma=sigma, K=K, alph = alph, delta = delta, B = B, m_covariance = m_covariance, r_variance = r_variance)
        prices_2.append( MC_price)
        lower_bounds_2.append(IC_bas)
        upper_bounds_2.append(IC_haut)


    # Tracer les résultats
    plt.figure(figsize=(10, 6))

    plt.plot(nb_simul_list, prices_1, label="Prix estimé de l'option P_DO sans variable de controle", color="blue", marker='o')
    plt.fill_between(nb_simul_list, lower_bounds_1, upper_bounds_1, 
                 color="blue", alpha=0.2, 
                 edgecolor="black", linewidth=1.5,
                 label="Intervalle de confiance 90% (méthode 1)")
    
    plt.plot(nb_simul_list, prices_2, label="Prix estimé de l'option P_DO avec variable de controle", color="green", marker='o')
    plt.fill_between(nb_simul_list, lower_bounds_2, upper_bounds_2, 
                 color="green", alpha=0.2, 
                 edgecolor="black", linewidth=1.5,
                 label="Intervalle de confiance 90% (méthode 2)")

    # Tracer les intervalles de confiance
   

    
    
 
    plt.title("Comparaison des estimations du prix d'une option barriere avec et sans variable de controle")
    plt.xlabel("Nombre de simulations")
    plt.ylabel("Prix estimé de l'option") 

    plt.legend()
    plt.grid(True)
    plt.show()






# ========================================
# Compare les intervalles de confiance pour la méthode VC
# en fonction de différentes valeurs de barrière B
#
# Entrées :
# - valeurs_B : liste des barrières à tester
# - autres paramètres standards
#
# Sortie : graphique des intervalles de confiance
# ========================================

def discriminant_VC (valeurs_B, nb_simul= 120000,  S0=1, r=0.015, T=2, sigma=0.15, K=1, alph=0.1, delta = 1/52, m_covariance = False, r_variance = True):

    """
    prices_1 = []
    lower_bounds_1 = []
    upper_bounds_1 = []
    """

    prices_2 = []
    lower_bounds_2 = []
    upper_bounds_2 = []

    for B in valeurs_B:

        """
        MC_price, (IC_bas, IC_haut) = calcul_P_DO(nb_simul, S0=S0, r=r, T=T, sigma=sigma, K=K, alph = alph, delta = delta, B = B, m_covariance = m_covariance, r_variance= r_variance)
        prices_1.append( MC_price)
        lower_bounds_1.append(IC_bas)
        upper_bounds_1.append(IC_haut)
        """

        MC_price, (IC_bas, IC_haut) = variable_controle( nb_simul=nb_simul, S0= S0, r=r, T=T, sigma=sigma, K=K, alph = alph, delta = delta, B = B, m_covariance = m_covariance, r_variance = r_variance)
        prices_2.append( MC_price)
        lower_bounds_2.append(IC_bas)
        upper_bounds_2.append(IC_haut)


    # Tracer les résultats
    plt.figure(figsize=(10, 6))
    
    """
    plt.plot(valeurs_B, prices_1, label="Intervalle confiance du prix estimé de l'option P_DO sans VC en fonction de B", color="blue", marker='o')
    plt.fill_between(valeurs_B, lower_bounds_1, upper_bounds_1, 
                 color="blue", alpha=0.2, 
                 edgecolor="black", linewidth=1.5,
                 label="Intervalle de confiance 90% (méthode 1)")
    """
    
    plt.plot(valeurs_B, prices_2, label="Intervalle confiance du prix estimé de l'option P_DO avec VC en fonction de B", color="green", marker='o')
    plt.fill_between(valeurs_B, lower_bounds_2, upper_bounds_2, 
                 color="green", alpha=0.2, 
                 edgecolor="black", linewidth=1.5,
                 label="Intervalle de confiance 90% (méthode 2)")

    # Tracer les intervalles de confiance
   

    
    
 
    plt.title("Comparaison des intervalles de confiance des estimations du prix d'une option barriere avec et sans VC en fonction de B")
    plt.xlabel("valeurs de B")
    plt.ylabel("Prix estimé de l'option") 

    plt.legend()
    plt.grid(True)
    plt.show()