from fonctions import *
from MC_file import *



"""
    Fonction principale exécutant le menu interactif de simulations d'options.

    - Affiche un menu de 17 choix différents pour simuler/prix des options européennes (vanille ou à barrière).
    - Paramètres globaux prédéfinis pour les différentes méthodes de Monte Carlo.
    - En fonction du choix utilisateur, appelle la fonction appropriée avec les bons paramètres.

    Entrée : aucune (interactivité via input)
    Sortie : aucune (affiche les résultats ou les graphiques correspondants)
"""

def main():
    
    ### Proposition des differents choix possibles
    print("""
    Menu des simulations:
    0 - Option européenne vanille sans barrière
    1 - Option barrière sans réduction de variance (méthode covariance)
    2 - Option barrière sans réduction de variance (méthode log_spath)
    3 - Option barrière avec réduction de variance
    4 - Comparaison avec/sans réduction de variance
    5 - Option barrière en fonction de B
    6 - Option barrière en fonction de sigma (S0=1)
    7 - Option barrière en fonction de sigma (S0=0.8)
    8 - Comparaison de l'option barrière en fonction de sigma pour S0 = 1 et 0.8
    9 - Option barriere P_DO approximé en fonction du nombre de trajectoires
    10 - Comparaison option barriere sous P_DO_delta et P_DO en fonction de delta
    11 - Comparaison probabilité de non-sortie sous ksi_DO_delta et ksi_DO en fonction de delta
    12 - Option barriere P_DO avec variable de controle en fonction du nombre de trajectoires
    13 - Comparaison option barriere P_DO avec et sans variable de controle en fonction du nombre de trajectoires
    14 - Comparaison des intervalles de confiance des estimations du prix d'une option barriere avec et sans VC en fonction de B
    15 - Simuler les trajectoires des browniens et répresenter la loi gaussienne W
    16 - Simuler les trajectoires du cours de l'actif avec B, K et S0
    """)

    
    ### Choix de la configuration
    choix = input("Entrez votre choix (0-16): ")
    print("\n")
    
    ### Paramètres communs
    nb_simul_list = [1000, 3000, 5000, 8000, 13000, 25000, 40000, 65000, 120000, 500000] 
    #nb_simul_list = [1000, 3000, 5000, 8000, 13000, 25000] 
    nb_simul_precis = 120000
    valeurs_B = np.linspace(0.5, 1, 15)
    valeurs_sigma = np.linspace(0, 0.8, 15)
    valeurs_delta = [3, 1, 1/4, 1/12, 1/52, 1/100, 1/250, 1/275, 1/300]
    

    ### Execution du choix

    if choix == "0":
        # Option européenne vanille sans barriere
        calcul_P_trajectoires(nb_simul_list, barriere=False)

    elif choix == "1":
        # Option barriere sans réduction de variance avec methode de covariance
        calcul_P_trajectoires(nb_simul_list, barriere=True, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph=0.1, delta = 1/52 ,B = 0.7, m_covariance = True, r_variance = False, comparer_r_var = False)

    elif choix == "2":
        # Option barriere sans réduction de variance avec methode de log_spath
        calcul_P_trajectoires(nb_simul_list, barriere=True, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph=0.1, delta = 1/52 ,B = 0.7, m_covariance = False, r_variance = False, comparer_r_var = False)
        
    elif choix == "3":
        # Option barriere avec réduction de variance
        calcul_P_trajectoires(nb_simul_list, barriere=True, r_variance=True, comparer_r_var=False)
    
    elif choix == "4":
        # Comparaison avec/sans réduction de variance
        calcul_P_trajectoires(nb_simul_list, barriere=True, r_variance=True, comparer_r_var=True)
        
    elif choix == "5":
        # Option barrière en fonction de B
        comparer_B(valeurs_B, nb_simul=nb_simul_precis)
        
    elif choix == "6":
        # Option barrière en fonction de sigma (S0=1)
        comparer_sigma(valeurs_sigma, nb_simul=nb_simul_precis, S0=1, comparer=False)
        
    elif choix == "7":
        # Option barrière en fonction de sigma (S0=0.8)
        comparer_sigma(valeurs_sigma, nb_simul=nb_simul_precis, S0=0.8, comparer=False)
        
    elif choix == "8":
        # Comparaison pour différents S0 (1 et 0.8)
        comparer_sigma(valeurs_sigma, nb_simul=nb_simul_precis, S0=1, comparer=True)

    elif choix == '9':
        # Option barriere P_DO approximé en fonction du nombre de trajectoires
        calcul_P_DO_trajectoires(nb_simul_list, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph=0.1, delta = 1/52 ,B = 0.7, m_covariance = False, r_variance = True)
         
    elif choix == '10':
        # Comparaison option barriere sous P_DO_delta et P_DO en fonction de delta
        comparer_DO_et_DO_delta(valeurs_delta = valeurs_delta, delta_DO = 1/100 , nb_simul = nb_simul_precis, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph=0.1, B = 0.7, m_covariance = False, r_variance = True)  

    elif choix =='11':
        # Comparaison de probabilité de non-sortie sous ksi_DO_delta et ksi_DO en fonction de delta
        comparer_proba_non_sortie_delta(valeurs_delta = valeurs_delta, delta_DO= 1/100 , nb_simul = nb_simul_precis, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph = 0.1, B = 0.7, m_covariance = False, r_variance = False)

    elif choix =='12':
        # Option barriere P_DO avec variable de controle en fonction du nombre de trajectoires
        calcul_P_DO_VC_trajectoires(nb_simul_list, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph=0.1, delta = 1/100 ,B = 0.7, m_covariance = False, r_variance = True)

    elif choix =='13':
        # Comparaison option barriere P_DO avec et sans variable de controle en fonction du nombre de trajectoires
        comparer_VC_et_sans_VC(nb_simul_list, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph=0.1, delta = 1/100 ,B = 0.7, m_covariance = False, r_variance = False)

    elif choix =='14':
        # Comparaison des intervalles de confiance des estimations du prix d'une option barriere avec et sans VC en fonction de B
        discriminant_VC (valeurs_B, nb_simul= 25000,  S0=1, r=0.015, T=2, sigma=0.15, K=1, alph=0.1, delta = 1/100, m_covariance = False, r_variance = True)
    
    elif choix =='15':
        # Simuler une loi gaussienne densité par Box-Muller et méthode de Cholesky
        resultats = simuler_W(1000000, T=2, barriere = False, delta = 1/52)
        plot_graph_gaussienne(resultats)

        # Simuler les trajectoires des browniens
        plot_trajectoires_browniennes(nb_simul=50, T=2, delta=1/52)

    elif choix =='16':
        # Simuler les trajectoires du cours de l'actif avec B, K et S0
        tracer_trajectoires_st(nb_simul=100, S0=1, K=1, r=0.015, sigma=0.15, T=2, delta=1/52, B= 0.4, m_covariance=True)
        tracer_trajectoires_st(nb_simul=100, S0=1, K=1, r=0.015, sigma=0.15, T=2, delta=1/52, B= 0.8, m_covariance=True)

    else:
        print("Choix invalide. Veuillez entrer un nombre entre 0 et 16.")
        




if __name__ == "__main__":

    """
    Point d'entrée du script.
    Exécute la boucle principale tant que l'utilisateur choisit 'Y' après chaque simulation.
    """
     
    Boucle = True
    while Boucle :
        main()
        response = input("Touch Y to continue or any others letters to exit : ")
        if response.lower() != 'y':
           Boucle = False
