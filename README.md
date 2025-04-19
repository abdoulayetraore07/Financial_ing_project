# 💹 Financial Engineering Project: Barrier Options

## 📌 Description
Ce projet implémente différentes méthodes de calcul du prix d'options à barrière dans le cadre du modèle de Black-Scholes. Développé pour le cours PRB209, ce travail explore à la fois les solutions analytiques et les méthodes de simulation Monte Carlo pour la valorisation d'options européennes classiques et d'options à barrière (Down & Out, Down & In).

Le projet démontre l'influence de la discrétisation des observations et l'utilisation de techniques de réduction de variance pour améliorer la précision des estimations.

---

## 🛠️ Fonctionnalités
✅ **Valorisation analytique** - Calcul exact des prix d'options selon les formules de Black-Scholes  
✅ **Simulation Monte Carlo** - Implémentation de la méthode classique et avec réduction de variance  
✅ **Options à barrière** - Support des options Down & Out et Down & In  
✅ **Intervalles de confiance** - Calcul et visualisation des intervalles de confiance à 90%  
✅ **Analyse de sensibilité** - Étude de l'impact des paramètres clés (volatilité, barrière, pas de discretisation,  etc.)  
✅ **Analyse de la porbabilité de non-sortie** - Étude de l'impact du pas de discretisation sur la proba de non-franchissement de la barrière 
✅ **Approximation d'Abramowitz & Stegun** - Implémentation de l'approximation pour la fonction de distribution normale  

---

## 📂 Structure du Projet

```
/Financial_ing_project
│── README.md                      # Ce fichier
│── main.py                        # Point d'entrée principal du programme
│── MC_file.py                     # Implémentation des simulations Monte Carlo
│── fonctions.py                   # Fonctions utilitaires et calculs analytiques
```

## 🧠 Modèles et Méthodes Implémentés

### Options Européennes
- Calcul analytique du prix d'une option européenne vanille selon le modèle de Black-Scholes
- Simulation Monte Carlo pour le prix avec intervalles de confiance asymptotiques

### Options à Barrière Down & Out
Une option qui confère à son possesseur le droit de vendre à une date fixée T un actif à un prix K, uniquement si le prix de l'actif n'est pas descendu en dessous d'une barrière B avant T.

### Options à Barrière Down & In
Une option qui confère à son possesseur le droit de vendre à une date fixée T un actif à un prix K, uniquement si le prix de l'actif est descendu en dessous d'une barrière B avant T.

### Techniques de Réduction de Variance
- Méthode des variables antithétiques
- Méthode des variables de contrôle
- Analyse de l'impact sur la précision des estimations et les intervalles de confiance

## 🔢 Paramètres du Modèle
- **Volatilité (σ)** : 0.15 (défaut), plage d'étude [0, 0.8]
- **Taux d'intérêt (r)** : 0.015 (annuel)
- **Échéance (T)** : 2 ans
- **Prix d'exercice (K)** : 1
- **Valeur initiale (S₀)** : 1 (défaut), 0.8 pour certaines analyses
- **Barrière (B)** : 0.7 (défaut), plage d'étude [0.5, 1]
- **Discrétisation (Δ)** : Valeurs testées {1/250, 1/52, 1/12, 1/4, 1, 3}

## 🚀 Exécution du Projet
Pour exécuter le projet, lancez simplement `main.py` et choisissez la simulation souhaitée :
```bash
python main.py
