# 💹 Financial Engineering Project: Barrier Options

## 📌 Description
This project implements different pricing methods for barrier options within the Black-Scholes model. Developed for the PRB209 course, this work explores both analytical solutions and Monte Carlo simulation methods for pricing standard European options and barrier options (Down & Out, Down & In).

The project demonstrates the influence of observation discretization and the use of variance reduction techniques to improve estimation accuracy.

---

## Features
 **Analytical pricing** – Exact computation of option prices using Black-Scholes formulas  
 **Monte Carlo simulation** – Implementation of classical Monte Carlo and variance reduction methods  
 **Barrier options** – Support for Down & Out and Down & In options  
 **Confidence intervals** – Computation and visualization of 90% confidence intervals  
 **Sensitivity analysis** – Study of the impact of key parameters (volatility, barrier, discretization step, etc.)  
 **No-knockout probability analysis** – Study of the impact of discretization step on the probability of not hitting the barrier  
 **Abramowitz & Stegun approximation** – Implementation of the approximation for the normal distribution function  

---

## Project Structure

```
/Financial_ing_project
│── README.md                      # Ce fichier
│── main.py                        # Point d'entrée principal du programme
│── MC_file.py                     # Implémentation des simulations Monte Carlo
│── fonctions.py                   # Fonctions utilitaires et calculs analytiques
```


## Implemented Models and Methods

### European Options
- Analytical pricing of a vanilla European option using the Black-Scholes model
- Monte Carlo simulation for pricing with asymptotic confidence intervals

### Down & Out Barrier Options
An option that gives the holder the right to sell an asset at a strike price K at maturity T, only if the asset's price has not fallen below a barrier B before T.

### Down & In Barrier Options
An option that gives the holder the right to sell an asset at a strike price K at maturity T, only if the asset's price has fallen below a barrier B before T.

### Variance Reduction Techniques
- Antithetic variates method  
- Control variates method  
- Analysis of their impact on estimation accuracy and confidence intervals

## Model Parameters
- **Volatility (σ)**: 0.15 (default), study range [0, 0.8]  
- **Interest rate (r)**: 0.015 (annual)  
- **Maturity (T)**: 2 years  
- **Strike price (K)**: 1  
- **Initial value (S₀)**: 1 (default), 0.8 for certain analyses  
- **Barrier (B)**: 0.7 (default), study range [0.5, 1]  
- **Discretization (Δ)**: Tested values {1/250, 1/52, 1/12, 1/4, 1, 3}  

## Running the Project
To run the project, simply execute `main.py` and choose the desired simulation:
```bash
python main.py
