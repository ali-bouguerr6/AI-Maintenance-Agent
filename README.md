# AI Maintenance Agent ⚙️

AI Maintenance Agent is a fully local AI-based system designed to assist industrial maintenance teams. The project demonstrates how artificial intelligence can be used to analyze machine data, detect anomalies,anticipate failures, and support decision-making in an industrial context.


##  Project Context

In the industrial sector, machines are at the core of production processes. Unexpected machine failures can lead to production downtime, delivery delays, and high operational costs.

This project aims to illustrate how an AI agent can help monitor machine health, identify risky situations, and recommend preventive maintenance actions, while remaining fully local and secure.


## Objectives

The main objective is to build an AI agent capable of:

- Analyzing machine data
- Detecting anomalies using business rules
- Computing a risk score for each machine
- Prioritizing maintenance actions (P1 / P2 / P3)
- Interacting with users through natural language
- Using internal documents via a RAG system


##  Key Features

- KPI computation (machine failure rate, temperature, tool wear, etc.)
- Anomaly detection based on business rules
- Risk score calculation and machine classification (safe / warning / danger)
- Retrieval-Augmented Generation (RAG) using internal documents
- Conversational CLI interface
- Session-based conversational memory
- 100% local execution (no cloud, no external APIs)

## 🔄 Pipeline Overview

1. Load machine dataset
2. Compute industrial KPIs
3. Detect anomalies using business rules
4. Calculate a risk score per machine
5. Classify machines by risk level
6. Retrieve relevant internal documents (RAG)
7. Generate explanations and recommendations via a local LLM

## Démo 

## Installation & Execution

- Python 3.10+
- Ollama (local)

# Install dependencies 
```bash
pip install -r requirements.txt
```

## Run the full pipeline
```bash
python main.py
```

## Run the chat interface
```bash
python main_chat.py
```







