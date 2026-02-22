# 🔄 Guide de Mise à Jour depuis GitHub

Comment récupérer la dernière version du code sur une autre machine.

---

## 🚀 Scenario 1: Première fois (clonage initial)

### Commande:
```bash
git clone https://github.com/thchateau/tcopil2026.git
cd tcopil2026
```

### Résultat:
- ✅ Tout le projet téléchargé localement
- ✅ Branche `main` active par défaut
- ✅ Tous les fichiers prêts à l'emploi

---

## 🔄 Scenario 2: Mettre à jour une copie existante

Vous avez déjà cloné le repo et voulez les dernières modifications.

### Commande simple:
```bash
cd /chemin/vers/tcopil2026
git pull origin main
```

### Expliqué étape par étape:
```bash
# 1. Aller dans le dossier du projet
cd ~/projects/tcopil2026
# OU sur Windows:
cd C:\Users\VotreNom\projects\tcopil2026

# 2. Récupérer les changements du serveur
git pull origin main
```

### Résultat:
```
remote: Counting objects: 7, done.
remote: Compressing objects: 100% (7/7), done.
Unpacking objects: 100% (7/7), done.
From https://github.com/thchateau/tcopil2026
 * branch            main       -> FETCH_HEAD
   eef87a1..e38eff2  main       -> origin/main
Updating eef87a1..e38eff2
Fast-forward
 streamlit_predict_multi.py      |  10 +-
 PADDING_FEATURE.md              |   8 +-
 QUICK_START_PADDING.md          |   2 +-
 PADDING_POSITION.md             | 250 ++
 CHANGELOG_PADDING.md            |   5 +-
 5 files changed, 265 insertions(+), 10 deletions(-)
 create mode 100644 PADDING_POSITION.md
```

---

## 📊 Vérifier les changements reçus

Après un `git pull`, vous pouvez vérifier ce qui a changé:

### Voir le dernier commit:
```bash
git log -1 --oneline
```

Résultat:
```
e38eff2 Improve padding strategy: use end-of-sequence padding instead of beginning
```

### Voir tous les commits récents:
```bash
git log --oneline -10
```

### Voir les fichiers modifiés:
```bash
git diff HEAD~1 HEAD --name-only
```

### Voir les changements détaillés:
```bash
git diff HEAD~1 HEAD
```

---

## ⚙️ Configuration pour la première utilisation

Si c'est votre première fois avec ce repo, configurez Git:

```bash
# Configurer votre nom (visible dans les commits)
git config user.name "Votre Nom"

# Configurer votre email (visible dans les commits)
git config user.email "votre.email@example.com"

# Optionnel: Voir la configuration
git config --list
```

---

## 🔐 Si vous avez des modifications locales

**Scenario:** Vous avez modifié des fichiers localement et `git pull` fail

### Option 1: Garder vos modifications (recommandé)
```bash
# Sauvegarder vos changements
git stash

# Récupérer les mises à jour
git pull origin main

# Restaurer vos changements (peut créer des conflits)
git stash pop
```

### Option 2: Abandonner vos modifications
```bash
# ⚠️ ATTENTION: Cela supprime vos changements locaux!
git reset --hard origin/main
```

### Option 3: Faire un commit avant de pull
```bash
# Committer vos changements locaux
git add .
git commit -m "Mes modifications locales"

# Puis pull
git pull origin main
```

---

## 📍 Vérifier votre connexion au serveur

Avant de mettre à jour, vérifiez la connexion:

### Voir le remote configuré:
```bash
git remote -v
```

Résultat attendu:
```
origin  https://github.com/thchateau/tcopil2026.git (fetch)
origin  https://github.com/thchateau/tcopil2026.git (push)
```

### Tester la connexion:
```bash
git fetch origin --dry-run
```

Si pas d'erreur = ✅ OK!

---

## 🔄 Procédure complète d'update (recommandée)

Pour une mise à jour sûre et vérifiée:

```bash
# 1. Aller dans le dossier
cd ~/projects/tcopil2026

# 2. Vérifier le statut actuel
git status

# 3. Récupérer les changements
git pull origin main

# 4. Vérifier les changements reçus
git log -1 --oneline

# 5. Voir les fichiers modifiés
git diff HEAD~1 HEAD --stat

# 6. Afficher l'historique
git log --oneline -5
```

---

## 📦 Supprimer l'ancien cache après update

Important: Après une mise à jour du code de padding, supprimez le cache:

```bash
# Supprimer les données en cache (données processées)
rm -rf cached_datasets/

# Cela force à rétraiter les données au prochain run
```

---

## 🐛 Troubleshooting

### Erreur: "fatal: not a git repository"

**Cause:** Vous n'êtes pas dans un dossier Git

**Solution:**
```bash
cd /chemin/correct/vers/tcopil2026
git pull origin main
```

---

### Erreur: "error: Your local changes would be overwritten by merge"

**Cause:** Modifications locales en conflit

**Solution:**
```bash
# Option 1: Sauvegarder et pull
git stash
git pull origin main

# Option 2: Voir les conflits
git status
```

---

### Erreur: "fatal: Could not read from remote repository"

**Cause:** Problème d'authentification ou réseau

**Solution:**
```bash
# Vérifier la connexion internet
ping github.com

# Essayer à nouveau
git pull origin main

# Si ça persiste, vérifier les credentials GitHub
# (parfois nécessaire si token expiré)
```

---

## 🌐 Méthodes alternatives

### Via ligne de commande (curl):
```bash
# Télécharger le dernier code en ZIP
curl -L https://github.com/thchateau/tcopil2026/archive/refs/heads/main.zip -o tcopil2026.zip

# Extraire
unzip tcopil2026.zip
cd tcopil2026-main
```

### Via GitHub Desktop (interface graphique):
1. Ouvrir GitHub Desktop
2. File → Clone repository
3. Entrer: `https://github.com/thchateau/tcopil2026.git`
4. Cliquer "Clone"
5. Pour updater: Cliquer "Fetch origin"

---

## 📋 Checklist après update

Après avoir mis à jour, vérifiez:

- [ ] Fichiers téléchargés avec succès
- [ ] Pas d'erreur Git
- [ ] Cache supprimé (`rm -rf cached_datasets/`)
- [ ] Code compile correctement
- [ ] Tests passent (si applicable)

---

## 📊 Comparaison des versions

Pour voir les différences entre versions:

```bash
# Entre version locale et distante
git diff main origin/main

# Entre deux commits spécifiques
git diff eef87a1 e38eff2

# Entre branche et main
git diff HEAD origin/main -- streamlit_predict_multi.py
```

---

## 🔔 Récapitulatif des commandes essentielles

| Action | Commande |
|--------|----------|
| **Cloner** | `git clone https://github.com/thchateau/tcopil2026.git` |
| **Mettre à jour** | `git pull origin main` |
| **Voir statut** | `git status` |
| **Voir l'historique** | `git log --oneline -10` |
| **Voir changements** | `git diff HEAD~1 HEAD` |
| **Sauvegarder locales** | `git stash` |
| **Restaurer locales** | `git stash pop` |
| **Vérifier remotes** | `git remote -v` |

---

## 💡 Pro Tips

### Automatiser les mises à jour:
```bash
# Script shell (Linux/Mac)
#!/bin/bash
cd ~/projects/tcopil2026
git pull origin main
rm -rf cached_datasets/
echo "✅ Updated!"
```

### Créer un alias pour simplifier:
```bash
# Ajouter à ~/.bashrc (Linux/Mac) ou ~/.zshrc
alias update-tcopil="cd ~/projects/tcopil2026 && git pull origin main && rm -rf cached_datasets/"

# Utiliser:
update-tcopil
```

### Notification d'updates:
```bash
# Vérifier s'il y a des mises à jour sans pull
git fetch origin
git log main..origin/main --oneline
```

---

## ❓ Questions fréquentes

### Q: À quelle fréquence les mises à jour arrivent?
**A:** Selon les changements. Vérifiez `git log origin/main --oneline`.

### Q: Puis-je avoir plusieurs versions en local?
**A:** Oui, utilisez des branches: `git checkout -b ma-branche`

### Q: Peux-je revenir à une version antérieure?
**A:** Oui: `git checkout <commit-hash>`

### Q: Comment contribuer mes modifications?
**A:** Créez un fork, modifiez, puis faites un Pull Request sur GitHub.

---

**Version:** 2026-02-22  
**Git Version minimum:** 2.0+  
**Dernière mise à jour:** e38eff2
