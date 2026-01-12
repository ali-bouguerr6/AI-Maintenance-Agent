import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def ollama_generate(prompt: str, model: str = "qwen2.5:1.5b") -> str:
    """
    Appelle Ollama en local et renvoie le texte.
    Fallback automatique si RAM insuffisante :
    - qwen2.5:1.5b -> qwen2.5:0.5b
    - num_predict 400 -> 350
    """
    models_to_try = [model]
    if model != "qwen2.5:0.5b":
        models_to_try.append("qwen2.5:0.5b")

    attempts = [
        {"temperature": 0.2, "num_predict": 650},
        {"temperature": 0.2, "num_predict": 350},
    ]

    last_error = None

    for m in models_to_try:
        for opts in attempts:
            payload = {
                "model": m,
                "prompt": prompt,
                "stream": False,
                "options": opts
            }

            try:
                r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            except requests.RequestException as e:
                raise RuntimeError(
                    "Impossible de contacter Ollama sur http://localhost:11434. "
                    "Vérifie que 'ollama serve' tourne dans un autre terminal.\n"
                    f"Détail: {e}"
                )

            if r.status_code == 200:
                return r.json().get("response", "").strip()

            try:
                err = r.json()
            except Exception:
                err = {"error": r.text}

            last_error = f"Ollama error ({r.status_code}) avec modèle={m}, options={opts}: {err}"

            err_str = str(err).lower()
            if "more system memory" in err_str or "too large for system memory" in err_str:
                continue

            break

    raise RuntimeError(last_error)


def build_maintenance_prompt(kpis: dict, risk_summary: dict, top_risky: list, risk_distribution: dict) -> str:
    context = {
        "kpis": kpis,
        "risk_summary": risk_summary,
        "risk_distribution": risk_distribution,
        "top_risky_machines": top_risky
    }

    context_json = json.dumps(context, ensure_ascii=False)

    return (
        "Tu es un ingénieur maintenance industrielle.\n"
        "Tu dois produire un rapport STRICTEMENT basé sur le JSON fourni.\n\n"
        "RÈGLES ANTI-HALLUCINATION (OBLIGATOIRES):\n"
        "1) Ne JAMAIS inventer d'ID machine. Tu dois recopier exactement les 'machine_id' présents dans top_risky_machines.\n"
        "2) Ne JAMAIS inventer de norme/seuil. Interdit: 'norme', 'seuil', 'trop haut', 'trop bas', '< 305K', 'exceeding the norm'.\n"
        "3) Ne JAMAIS parler d'augmentation, baisse, tendance, évolution (car aucune série temporelle n'est fournie).\n"
        "4) Le champ 'signals' est une LISTE contenant uniquement: *_anomaly et/ou 'machine_failure'.\n"
        "   - Interdit de mettre des variables dedans (ex: 'tool_wear_min', 'rpm', 'process_temp_k').\n"
        "   - Si signals est vide: écrire exactement 'signals: []'.\n"
        "5) Les KPI (kpis) sont des moyennes globales. Les valeurs machine sont uniquement dans top_risky_machines.\n\n"
        "FORMAT OBLIGATOIRE (respecte exactement les titres):\n"
        "SYNTHÈSE:\n"
        "- 3 à 6 lignes max, factuel.\n\n"
        "KPI:\n"
        "- 5 à 8 puces max (reprendre les valeurs de kpis).\n\n"
        "RISQUE:\n"
        "- min/mean/max (depuis risk_summary)\n"
        "- répartition safe/warning/danger (depuis risk_distribution)\n\n"
        "TOP10:\n"
        "- 10 lignes EXACTEMENT, une par machine, au format:\n"
        "  machine_id | risk_score | signals | observation (1 phrase prudente)\n"
        "- IMPORTANT: machine_id doit être recopié exactement.\n\n"
        "RECOMMANDATIONS:\n"
        "- P1: 2 actions concrètes (priorité haute)\n"
        "- P2: 2 actions concrètes (priorité moyenne)\n"
        "- P3: 2 actions concrètes (priorité basse)\n\n"
        f"JSON:\n{context_json}"
    )
