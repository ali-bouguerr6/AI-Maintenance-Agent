
#  AI Maintenance Agent

AI Maintenance Agent est un projet d’agent intelligent **100 % local** dédié à la **maintenance industrielle prédictive**.  
Il combine l’analyse de données machines, la détection d’anomalies, l’évaluation des risques et l’explication en langage naturel grâce à l’intelligence artificielle.

L’objectif est d’aider à **anticiper les pannes**, **réduire les arrêts machines** et **faciliter la prise de décision** pour les équipes de maintenance.

---

##  Objectifs du projet
- Exploiter des données industrielles pour surveiller l’état des machines  
- Détecter automatiquement des anomalies  
- Calculer un score de risque de défaillance  
- Fournir des explications compréhensibles via un agent IA  
- Fonctionner sans cloud, en environnement local  

---

##  Fonctionnalités principales
- Calcul de KPI de maintenance  
- Détection d’anomalies basée sur des règles métier  
- Évaluation du niveau de risque (faible / moyen / élevé)  
- Génération d’explications en langage naturel (LLM local)  
- RAG (Retrieval Augmented Generation) sur documents techniques  
- Mode pipeline et mode chat  


##  Données utilisées
Le projet repose sur un dataset de **maintenance prédictive** (type AI4I 2020) contenant :
- températures
- vitesse de rotation
- couple
- usure des outils
- indicateurs de défaillance

Ces données permettent de simuler un environnement industriel réaliste.

---

## AI & RAG
- Utilisation d’un LLM local  
- Système RAG pour interroger :
  - procédures de maintenance
  - politiques de sécurité
  - fiches machines  
- Aucune dépendance au cloud → confidentialité des données  

---

## ▶️ Utilisation

### Lancer le pipeline principal
```bash
python main.py
```

### Lancer l’agent en mode chat
```
python main_chat.py
```

###Indexer les documents (RAG)
```
python rag/ingest.py
```
###Interroger les documents
```
python rag/query.py
```






