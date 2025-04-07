from fonctions import *

def main():

    # Liste des tailles de simulations
    nb_simul_list = [1000, 3000, 5000, 8000, 13000, 25000, 40000, 65000, 90000, 120000, 1000000]

    # Appeler la fonction pour afficher le graphique
    calcul_P_euro_trajectoires(nb_simul_list, barriere = False, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph=0.1, delta = 1/52 ,B = 0.7, m_covariance = False) 
    calcul_P_euro_trajectoires(nb_simul_list, barriere = True, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph=0.1, delta = 1/52 ,B = 0.7, m_covariance = True)  
    calcul_P_euro_trajectoires(nb_simul_list, barriere = True, S0=1, r=0.015, T=2, sigma=0.15, K=1, alph=0.1, delta = 1/52 ,B = 0.7, m_covariance = False)  
      

if __name__ == "__main__":
    main()