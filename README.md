# 🔮 ML Predictor : 4 modèles de prédiction (Streamlit)

| Modèle | Dataset | Type | Cible | Meilleur algo |
|---|---|---|---|---|
| 📱 Téléphones | Telephone_data.csv | Classification | `etat` | Random Forest |
| 🚗 Voitures (état) | Car_Data.csv | Classification | `Venant_Occasion` | Random Forest |
| 💰 Voitures (prix) | Car_dataset.csv | Régression | `Selling_Price` | Random Forest |
| 🚕 Taxi | taxi_trip_pricing.csv | Régression | `Trip_Price` | Random Forest |

Chaque modèle a 2 onglets dans l'application : **Formulaire** (une prédiction) et **Fichier CSV** (un tableau de prédictions, téléchargeable).

## Structure
```
app.py                          # application Streamlit
utils.py                        # prétraitement (IQRClipper...) : nécessaire pour charger les .pkl
models/*.pkl                    # les 4 modèles entraînés
Entrainement_4_modeles.ipynb    # notebook d'entraînement (Colab ou local)
exemples_csv/                   # CSV d'exemple pour tester l'import
requirements.txt, .streamlit/config.toml
```

## Lancer en local (PowerShell)
```powershell
cd projet_ml
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```
Si PowerShell bloque l'activation : `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Déployer sur Streamlit Community Cloud (gratuit)
1. Créer un dépôt GitHub **public** et y pousser le dossier :
```powershell
git init
git add .
git commit -m "ML Predictor"
git branch -M main
git remote add origin https://github.com/<ton-pseudo>/<ton-repo>.git
git push -u origin main
```
2. Aller sur https://share.streamlit.io, se connecter avec GitHub, **Create app**.
3. Choisir le dépôt, branche `main`, fichier principal `app.py`, puis **Deploy**.
4. Après 2 à 3 minutes, l'URL publique `https://<nom>.streamlit.app` est prête à partager.

## Réentraîner les modèles
Lancer le notebook (Colab : importer les 4 CSV d'abord), puis télécharger `models/` et `utils.py` et les remplacer dans le projet.
⚠️ La version de scikit-learn affichée au début du notebook doit être la même que dans `requirements.txt` (sinon erreur au chargement des `.pkl`).
