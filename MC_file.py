import math as mt
import numpy as np
import random
import scipy.stats as stats
from fonctions import *





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
        T_path = [(i+1)*delta for i in range(N_delta)]              ## On avait fait une erreur là car on multipliait par T tout simplement au lieu de T_path
        T_path =  np.broadcast_to(T_path , (nb_simul, N_delta))    ## Fonction pour repeter le meme vecteur sur chaque ligne (On get une matrice)
        S_path  = S0 * np.exp( (r - (sigma**2)/2) * T_path + sigma*gaussien_W_path )

        if r_variance:
            S_path_neg  = S0 * np.exp( (r - (sigma**2)/2) * T_path + sigma*(-gaussien_W_path) )

    #Conditionnement pour avoir une formule plus compacte 
    S_path_neg =  S_path_neg*(r_variance==True) + S_path*(r_variance==False)

    return S_path, S_path_neg





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




def comparer_delta_DO_et_DO_delta(valeurs_delta = [1/52], nb_simul = 120000, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph=0.1, B = 0.7, m_covariance = False, r_variance = True):
 
    prices_DO_delta = []
    prices_DO = []

    for delta in valeurs_delta:
        MC_price_DO_Delta, (IC_bas, IC_haut) = calcul_P_euro_et_DO_delta(nb_simul=nb_simul, barriere= True, S0=S0, r=r, T=T, sigma=sigma, K=K, alph = alph, delta = delta, B = B, m_covariance = m_covariance, r_variance= r_variance)
        prices_DO_delta.append(MC_price_DO_Delta)
        MC_price_DO, (IC_bas, IC_haut) = calcul_P_DO(nb_simul=nb_simul, S0=S0, r=r, T=T, sigma=sigma, K=K, alph = alph, delta = delta, B = B, m_covariance = m_covariance, r_variance = r_variance)
        prices_DO.append(MC_price_DO)
           
    plt.figure(figsize=(10, 6))

    plt.plot(valeurs_delta, prices_DO_delta, label="Prix estimé de l'option avec P_DO_delta en fonction de delta ", color="blue", marker='o')
    plt.plot(valeurs_delta,prices_DO, label="Prix estimé de l'option avec P_DO en fonction de delta", color="green", marker='o')
    
    plt.title("Estimations du prix d'une option barriere européenne vanille en fonction de delta avec reduction de variance")
    plt.xlabel("Valeurs de delta")
    plt.ylabel("Prix estimé de l'option") 


    plt.legend()
    plt.grid(True)
    plt.show()


def comparer_proba_non_sortie_delta(valeurs_delta = [1/52], nb_simul = 120000, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph = 0.1, B = 0.7, m_covariance = False, r_variance = False):

    Proba_in_DO_delta = []
    Proba_in_DO = []
    cpt = 1
    for delta in  valeurs_delta :
        #Obtention des trajectoires
        S_path, S_path_neg = get_S_path(nb_simul, S0=S0, r=r, T=T, sigma=sigma, delta = delta, m_covariance = m_covariance, r_variance = r_variance)

        #Obtention du vecteur des mins et vecteur pont gaussien sur chaque trajectoire
        S_path_min  = np.min(S_path, axis = 1)
        S_path_hit  = correction_discretisation(S_path=S_path, B=B, sigma=sigma, delta=delta)
    
        # Calcul des payoffs
        payoff_DO_delta = (S_path_min>=B) 
        payoff_DO = S_path_hit   

        #ESTIMATION GLOBLALE DU PRIX DE L'OPTION PAR MONTE-CARLO
        proba_in_DO_delta, (IC_bas, IC_haut) = compute_MC(payoff=payoff_DO_delta, nb_simul=nb_simul,  alph=alph, calcul_proba= True)
        proba_in_DO, (IC_bas, IC_haut) = compute_MC(payoff=payoff_DO, nb_simul=nb_simul,  alph=alph, calcul_proba= True)

        Proba_in_DO_delta.append(proba_in_DO_delta)
        Proba_in_DO.append(proba_in_DO)
        print(f"Etapes : {cpt} / {len(valeurs_delta)} terminées")
        cpt+=1

    print("\n")
    plt.figure(figsize=(10, 6))

    plt.plot(valeurs_delta,  Proba_in_DO_delta, label="Probabilité de non-sortie de ksi_DO_delta en fonction de delta", color="blue", marker='o')
    plt.plot(valeurs_delta, Proba_in_DO, label="Probabilité de non-sortie de ksi_DO en fonction de delta ", color="green", marker='o')
    
    plt.title("Probabilité de non-franchissement de barriere en fonction de delta")
    plt.xlabel("Valeurs de delta")
    plt.ylabel("Probabilité de non-franchissement") 


    plt.legend()
    plt.grid(True)
    plt.show()
    