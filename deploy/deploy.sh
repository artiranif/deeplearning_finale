#!/bin/bash
# =============================================================
#  deploy.sh — Déploiement automatique sur VPS
#  Usage : ./deploy/deploy.sh
# =============================================================
set -euo pipefail

DOMAIN="python-community.com"
API_SUBDOMAIN="api.${DOMAIN}"
MLFLOW_SUBDOMAIN="mlflow.${DOMAIN}"
APP_DIR="/opt/sonar-classifier"
REPO_URL="${REPO_URL:-}"

echo "══════════════════════════════════════════════════"
echo "  Déploiement Sonar Classifier sur VPS"
echo "══════════════════════════════════════════════════"

# ── 1. Mise à jour du code ──────────────────────────────────
if [ -d "$APP_DIR" ]; then
    echo "→ Mise à jour du code..."
    cd "$APP_DIR"
    git pull origin main
else
    echo "→ Clonage du repo..."
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

# ── 2. Première fois : obtenir les certificats SSL ──────────
if [ ! -d "/etc/letsencrypt/live/${API_SUBDOMAIN}" ]; then
    echo "→ Obtention des certificats SSL..."

    # Nginx temporaire pour le challenge ACME
    docker compose -f docker-compose.init-ssl.yml up -d nginx-init

    # Demande des certificats
    docker run --rm \
        -v sonar-classifier_certbot-data:/var/www/certbot \
        -v sonar-classifier_certbot-certs:/etc/letsencrypt \
        certbot/certbot certonly \
        --webroot --webroot-path=/var/www/certbot \
        --email admin@${DOMAIN} \
        --agree-tos --no-eff-email \
        -d ${API_SUBDOMAIN} \
        -d ${MLFLOW_SUBDOMAIN}

    docker compose -f docker-compose.init-ssl.yml down
    echo "✅ Certificats SSL obtenus"
fi

# ── 3. Build & déploiement ──────────────────────────────────
echo "→ Build des images Docker..."
docker compose build --no-cache

echo "→ Démarrage des services..."
docker compose up -d

echo "→ Nettoyage des anciennes images..."
docker image prune -f

# ── 4. Vérification ────────────────────────────────────────
echo ""
echo "→ Vérification des services..."
sleep 5

if docker compose ps | grep -q "Up"; then
    echo "✅ Services démarrés avec succès !"
    echo ""
    echo "  API     : https://${API_SUBDOMAIN}"
    echo "  MLflow  : https://${MLFLOW_SUBDOMAIN}"
    echo "  Health  : https://${API_SUBDOMAIN}/health"
    echo "  Docs    : https://${API_SUBDOMAIN}/docs"
else
    echo "❌ Erreur : certains services ne sont pas démarrés"
    docker compose logs --tail=50
    exit 1
fi
