# Sonar Classification — MLOps Pipeline

Pipeline MLOps complète pour la classification de signaux sonar (Mines vs Rochers) avec déploiement automatisé sur VPS.

## Architecture

```
┌──────────┐     ┌─────────────┐     ┌──────────┐     ┌────────┐     ┌─────────┐
│   DVC    │────▶│ scikit-learn │────▶│ FastAPI  │────▶│ GitHub │────▶│  Docker │
│ (données)│     │  (training)  │     │  (API)   │     │Actions │     │ (build) │
└──────────┘     └─────────────┘     └──────────┘     │(CI/CD) │     └────┬────┘
                       │                               └────────┘          │
                 ┌─────▼─────┐     ┌──────────────┐                  ┌────▼────┐
                 │  MLflow   │     │  Evidently   │                  │   VPS   │
                 │(tracking) │     │ (monitoring) │                  │(deploy) │
                 └───────────┘     └──────────────┘                  └─────────┘
```

## Stack technique

| Outil | Rôle |
|-------|------|
| **DVC** | Versionner données et pipelines |
| **scikit-learn** | Entraîner le modèle (RandomForest) |
| **MLflow** | Suivre les expériences et métriques |
| **FastAPI** | Exposer le modèle en API REST |
| **Evidently** | Monitorer le data drift et qualité |
| **Docker** | Empaqueter l'application |
| **GitHub Actions** | CI/CD automatisé |
| **Nginx + Let's Encrypt** | Reverse proxy + HTTPS |

## Structure du projet

```
├── .github/workflows/ci-cd.yml   # Pipeline CI/CD
├── .dvc/                          # Configuration DVC
├── app/
│   └── main.py                    # API FastAPI
├── src/
│   ├── train.py                   # Script d'entraînement
│   └── generate_data.py           # Génération de données test
├── data/
│   └── sonar_data.csv.dvc         # Données versionnées par DVC
├── models/                        # Artefacts modèle (model.pkl, scaler.pkl)
├── tests/
│   └── test_api.py                # Tests API
├── deploy/
│   ├── deploy.sh                  # Script de déploiement VPS
│   └── nginx/                     # Configuration Nginx
├── docker-compose.yml             # Stack complète
├── Dockerfile                     # Image API
├── dvc.yaml                       # Pipeline DVC
└── requirements.txt               # Dépendances Python
```

## Démarrage rapide

### 1. Installation locale

```bash
# Cloner le repo
git clone https://github.com/artiranif/deeplearning_finale.git
cd deeplearning_finale

# Environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Dépendances
pip install -r requirements.txt
```

### 2. Entraînement du modèle

```bash
# Via DVC pipeline
dvc repro

# Ou directement
python src/train.py
```

### 3. Lancer l'API localement

```bash
uvicorn app.main:app --reload --port 8000
```

API disponible sur : http://localhost:8000/docs

### 4. Docker local

```bash
docker build -t sonar-api .
docker run -p 8000:8000 sonar-api
```

## Endpoints API

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/` | Statut de l'API |
| GET | `/health` | Health check |
| GET | `/docs` | Documentation Swagger |
| POST | `/predict` | Prédiction avec features JSON |
| POST | `/predict/generated` | Prédiction avec données générées |
| POST | `/predict_file` | Prédiction depuis fichier CSV |
| POST | `/generate-data` | Générer des données aléatoires |
| GET | `/generated-data` | Voir les données générées |
| POST | `/monitor` | Monitoring drift (upload CSV) |
| POST | `/monitor/generated` | Monitoring avec données générées |
| GET | `/monitor/report` | Voir le rapport Evidently |

## Déploiement sur VPS

### Prérequis VPS

- Ubuntu avec Docker et Docker Compose installés
- Sous-domaines DNS configurés :
  - `api.python-community.com` → IP du VPS
  - `mlflow.python-community.com` → IP du VPS

### Secrets GitHub à configurer

Dans **Settings → Secrets and variables → Actions** du repo :

| Secret | Description |
|--------|-------------|
| `VPS_HOST` | Adresse IP du VPS |
| `VPS_USER` | Utilisateur SSH (ex: `root` ou `deploy`) |
| `VPS_SSH_KEY` | Clé privée SSH |

### Premier déploiement

```bash
# Sur le VPS
git clone https://github.com/artiranif/deeplearning_finale.git /opt/sonar-classifier
cd /opt/sonar-classifier

# Obtenir les certificats SSL
chmod +x deploy/deploy.sh
REPO_URL=https://github.com/artiranif/deeplearning_finale.git ./deploy/deploy.sh
```

### Déploiements suivants

Automatiques via GitHub Actions à chaque push sur `main` :

1. **Test** → pytest
2. **Build** → Docker image
3. **Deploy** → SSH vers VPS, charge l'image, redémarre les services

## URLs de production

- **API** : https://api.python-community.com
- **Docs API** : https://api.python-community.com/docs
- **MLflow** : https://mlflow.python-community.com
- **Monitoring** : https://api.python-community.com/monitor/report

## Monitoring avec Evidently

```bash
# Générer un rapport de drift
curl -X POST https://api.python-community.com/monitor/generated

# Voir le rapport HTML
open https://api.python-community.com/monitor/report
```

## Pipeline DVC

```bash
# Reproduire le pipeline complet
dvc repro

# Voir le DAG
dvc dag

# Pousser les données
dvc push
```

## MLflow

```bash
# Accéder à l'UI MLflow (local)
mlflow ui --backend-store-uri sqlite:///mlflow.db

# En production
# https://mlflow.python-community.com
```
