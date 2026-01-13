# 🧠 TP3: Convolutional Neural Networks & Computer Vision

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-blue.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**École Nationale Supérieure Polytechnique de Yaoundé (ENSPY)**  
Département de Génie Informatique - 5GI  
**Matière :** Intelligence Artificielle  
**Instructeur :** Dr. Louis Fippo Fitime

**Auteur :** TCHASSI DANIEL  
**Matricule :** 21P073

---

## 🎯 Objectifs d'Apprentissage

Ce projet met en pratique les concepts fondamentaux des Réseaux de Neurones Convolutifs (CNNs) dans le cadre du TP3. Les objectifs sont de :
- **Comprendre** les principes de convolution, de pooling et des architectures CNN.
- **Construire** et entraîner un CNN pour la classification d'images sur CIFAR-10.
- **Intégrer** et comprendre l'utilité des blocs résiduels (ResNets).
- **Appliquer** les CNNs à des tâches avancées comme le transfert de style neuronal.
- **Maîtriser** l'implémentation de ces concepts avec TensorFlow/Keras et le suivi d'expériences avec MLflow.

---

## 📁 Structure du Projet

Le projet a été restructuré pour suivre la logique des exercices du TP, en séparant clairement le code réutilisable, les scripts d'exercices et le rapport.

```text
.
├── .github/workflows/
│   └── run_exercises.yml      # Workflow CI/CD pour lancer les exercices
├── exercises/
│   ├── exercise_1_cnn.py      # Script pour l'exercice 1 (CNN simple)
│   ├── exercise_2_resnet.py   # Script pour l'exercice 2 (ResNet)
│   └── exercise_4_style_transfer.py # Script pour l'exercice 4 (Transfert de Style)
├── src/
│   ├── data_loader.py         # Chargement et pré-traitement des données
│   └── models.py              # Architectures des modèles (CNN, ResNet)
├── images/
│   ├── content/               # Image de contenu pour l'exercice 4
│   └── style/                 # Image de style pour l'exercice 4
├── .gitignore
├── README.md
├── requirements.txt
└── main.py                    # Point d'entrée pour exécuter les exercices
```

---

## 🚀 Contenu des Exercices du TP

### Partie 1 & 3 : Questions Théoriques
Les réponses aux questions conceptuelles (rôle de la convolution, du pooling, des ResNets, de la segmentation, etc.) sont à rédiger indépendamment du projet, comme spécifié.

### Exercice 1 : Architecture CNN Classique
- **Fichier :** `exercises/exercise_1_cnn.py`
- **Objectif :** Construire et entraîner un CNN simple mais efficace sur le jeu de données CIFAR-10. L'architecture est `Conv -> Pool -> Conv -> Pool -> Flatten -> Dense`.
- **Expérience MLflow :** `TP3-Exercise1-BasicCNN`

### Exercice 2 : Réseaux Résiduels (ResNets)
- **Fichier :** `exercises/exercise_2_resnet.py`
- **Objectif :** Implémenter une architecture plus profonde en utilisant des blocs résiduels (avec *skip connections*) pour surmonter le problème de la dégradation des gradients.
- **Expérience MLflow :** `TP3-Exercise2-ResNet`

### Exercice 4 : Transfert de Style Neuronal
- **Fichier :** `exercises/exercise_4_style_transfer.py`
- **Objectif :** Utiliser un CNN pré-entraîné (VGG16) pour séparer le contenu d'une image et le style d'une autre, puis les recombiner pour créer une nouvelle image artistique.
- **Expérience MLflow :** `TP3-Exercise4-StyleTransfer`

---

## 🛠️ Installation et Utilisation

### 1. Prérequis
- Python 3.10+
- Un environnement virtuel (recommandé)
- Accès à un serveur MLflow (local ou distant)

### 2. Installation
```bash
# Clonez le dépôt et naviguez dans le dossier
git clone <votre-url-de-repo> && cd <nom-du-repo>

# Créez et activez un environnement virtuel
python3 -m venv venv && source venv/bin/activate

# Installez les dépendances
pip install -r requirements.txt
```

### 3. Configuration de l'Environnement (`.env`)
Avant de lancer un exercice, vous devez configurer la connexion à votre serveur MLflow. Pour cela, créez un fichier nommé `.env` à la racine du projet. Ce fichier contiendra les variables d'environnement nécessaires.

Voici un exemple de contenu pour votre fichier `.env` :
```env
# Adresse de votre serveur MLflow (obligatoire)
MLFLOW_TRACKING_URI=http://localhost:5000

# ----- Authentification (si votre serveur MLflow est protégé) -----
# MLFLOW_TRACKING_USERNAME=votre_nom_utilisateur
# MLFLOW_TRACKING_PASSWORD=votre_mot_de_passe

# ----- Stockage des artefacts sur un serveur S3/MinIO (option avancée) -----
# MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
# AWS_ACCESS_KEY_ID=minioadmin
# AWS_SECRET_ACCESS_KEY=minioadmin
```
**Variables principales :**
- `MLFLOW_TRACKING_URI`: C'est l'URL de votre serveur MLflow. C'est la seule variable obligatoire.
- `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`: À utiliser uniquement si votre serveur MLflow requiert une authentification.
- `MLFLOW_S3_ENDPOINT_URL` et les clés `AWS_*`: Pour les utilisateurs avancés qui souhaitent stocker les artefacts (modèles, images) sur un service de stockage objet compatible S3 comme MinIO, au lieu du système de fichiers local.

### 4. Exécution des Exercices
Le script `main.py` est le point d'entrée central.
- **Lancer l'Exercice 1 (CNN) :**
  ```bash
  python3 main.py --exercise 1
  ```
- **Lancer l'Exercice 2 (ResNet) :**
  ```bash
  python3 main.py --exercise 2
  ```
- **Lancer l'Exercice 4 (Transfert de Style) :**
  *(Assurez-vous d'avoir placé vos images dans les dossiers `images/content` et `images/style`)*
  ```bash
  python3 main.py --exercise 4 --content images/content/votre_image.jpg --style images/style/votre_style.jpg
  ```

---

## 📊 Suivi des Expériences avec MLflow

Chaque exécution d'un exercice est enregistrée dans une expérience MLflow dédiée. Vous pouvez y visualiser :
- Les **paramètres** utilisés (nombre d'époques, taille du batch...).
- Les **métriques** d'entraînement et de validation (perte, précision...).
- Les **artefacts** générés, comme le résumé du modèle, le rapport de classification, ou les images issues du transfert de style.

Accédez à votre interface MLflow pour comparer les performances des modèles CNN et ResNet.

---

## 🤖 Intégration Continue (CI/CD)

Le workflow `.github/workflows/run_exercises.yml` est configuré pour lancer automatiquement les exercices de classification (1 et 2) à chaque `push` sur la branche `main`. Cela garantit que le code d'entraînement reste fonctionnel et reproductible.

---
## ⚖️ Licence

Ce projet est distribué sous la licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus d'informations.