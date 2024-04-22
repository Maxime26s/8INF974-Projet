# 8INF974 - Étude Comparative des Algorithmes DQN, Double DQN et Prioritized Experience Replay
Ce projet est une implémentation des algorithmes DQN (Deep Q-Network) et Double DQN, avec une option de Prioritized Replay. Il est conçu pour fonctionner avec plusieurs environnements Gymnasium à action discrète, fournissant une base pour l'entraînement et le test de modèles d'apprentissage par renforcement.

## Auteurs
- Maxime Simard (SIMM26050001)

## Implémentation
- Le modèle initial de DQN a été implémenté en suivant les instructions de l'article original de Mnih et al. (2015) : [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602). Le tutoriel de PyTorch sur l'apprentissage par renforcement a été utilisé comme référence pour l'implémentation : [Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).
- Le modèle de Double DQN et de Prioritized Experience Replay ont été implémenté en se basant du répertoire GitHub de LabML.ai : [Deep Q Networks (DQN)](https://nn.labml.ai/rl/dqn/index.html), puis adapté en suivant les instructions de l'article original de Van Hasselt et al. (2016) : [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) et de Schaul et al. (2016) : [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952).

## Installation
Pour installer le projet, commencez par installer les dépendances requises avec la commande suivante :

```
pip install -r requirements.txt
```

## Utilisation
Pour lancer et utiliser l'application, suivez ces étapes :

1. Démarrer l'application

    Ouvrez un terminal et naviguez jusqu'au dossier du projet. Lancez le script principal en utilisant Python :

    ```
    python ./src [args]
    ```

2. Choisir un mode de fonctionnement

    L'application supporte trois modes de fonctionnement : entraînement, test et benchmark. Utilisez les arguments appropriés pour sélectionner le mode :

   - Pour entraîner : `--mode train`
   - Pour tester : `--mode test`
   - Pour le benchmark : `--mode benchmark`

    Pour le mode `test`, vous devez spécifier le chemin du modèle à tester dans le script `__main__.py`. De plus, assurez-vous d'utiliser le même environnement que celui utilisé pour l'entraînement.

3. Configuration avancée

    Modifiez les hyperparamètres spécifiques à chaque environnement dans la classe DQNAgent pour ajuster la taille de mémoire, la taille du lot, le taux d'apprentissage, etc.

4. Visualisation des résultats

    Les graphiques de l'évolution des récompenses, de la durée des épisodes et des pertes moyennes sont générés pendant l'entraînement pour visualiser les progrès.

5. Arrêter l'application

    Pour terminer l'exécution et fermer proprement toutes les instances de l'environnement, interrompez l'exécution du script (par exemple avec Ctrl+C dans le terminal).

## Exemple
Voici un exemple de commande pour démarrer un entraînement sur l'environnement CartPole-v1 en utilisant le mode de rendu visuel :

```
python ./src --mode train --game CartPole-v1 --render_mode human
```

Pour tester un modèle entraîné :

```
python ./src --mode test --game CartPole-v1
```

Pour effectuer un benchmark entre différentes configurations sur plusieurs jeux :

```
python ./src --mode benchmark
```

## Visualisation des résultats
Pour visualiser les résultats d'un entraînement, il est possible d'utiliser le script `visualization.py`. Ce script permet de générer des graphiques à partir des fichiers `.csv` enregistrés pendant l'entraînement. Vous devez simplement les déplacés dans le dossier `results` pour qu'ils soient pris en compte et les renommer avec le pattern suivant : `{game}_{model}_{variation}.csv`.

Ensuite, vous pouvez lancer le script de visualisation :

```
python ./src/visualization.py
```

## Exemple de résultats
Vous pouvez trouver des exemples de `.csv` et de graphique dans les dossier suivants:
- `results` : contient les fichiers `.csv`
- `out` : contient les graphiques générés
- `out/data` : contient les données utilisées pour générer les graphiques

## Rapport
Le rapport du projet est le fichier `Rapport.pdf`.
