import json

from core.loader import load_dataset
from core.kpis import compute_kpis
from core.anomalies import detect_anomalies
from core.risk_score import compute_risk_score, classify_risk_levels
from core.explain_llm import build_maintenance_prompt, ollama_generate

MODEL_NAME = "qwen2.5:1.5b"
VERBOSE = True  # False si tu veux moins de logs


def main():
    print("\n==============================")
    print(" AI MAINTENANCE AGENT - PIPELINE")
    print("==============================\n")

    # 1) Load dataset
    df = load_dataset()
    print(f"[1] Dataset chargé ✅  | lignes={len(df)}  colonnes={len(df.columns)}")
    if VERBOSE:
        print(df.head(), "\n")

    # 2) KPI
    kpis = compute_kpis(df)
    print("[2] KPIs calculés ✅")
    print(json.dumps(kpis, indent=4, ensure_ascii=False), "\n")

    # 3) Anomalies
    anom = detect_anomalies(df)
    total_anomalies = int(anom["total_anomalies"].sum())
    print("[3] Anomalies détectées ✅")
    if VERBOSE:
        print("Aperçu anomalies:")
        print(anom.head(), "\n")
    print(f"Nombre total d'anomalies dans le dataset : {total_anomalies}\n")

    # 4) Risk score
    risk = compute_risk_score(df, anom)
    print("[4] Risk score calculé ✅")
    if VERBOSE:
        print("Aperçu risk score:")
        print(risk.head(), "\n")
    print(f"Nombre de lignes risk score : {len(risk)}\n")

    # ====== Construire un tableau complet (df + anom + risk_score) ======
    full = df.copy()

    # ajouter toutes les colonnes anomalies (mêmes index)
    for col in anom.columns:
        full[col] = anom[col].values

    # ajouter le risk_score (même index)
    full["risk_score"] = risk["risk_score"].values

    # créer un ID lisible pour LLM
    if "Product ID" in full.columns:
        full["machine_id"] = full["Product ID"].astype(str).apply(lambda x: f"M{x}")
    else:
        # fallback: index comme ID
        full["machine_id"] = full.index.astype(str).apply(lambda x: f"M{x}")

    # 5) Top 10 machines risquées (sur full)
    top_risk = full.sort_values("risk_score", ascending=False).head(10)
    print("[5] Top 10 machines risquées ✅")
    show_cols = [c for c in ["machine_id", "Type", "risk_score", "Machine failure"] if c in top_risk.columns]
    print(top_risk[show_cols], "\n")

    # 6) Statistiques globales du risk score
    print("[6] Statistiques globales du risk score ✅")
    print(full["risk_score"].describe(), "\n")

    # 7) Classification safe / warning / danger
    risk_levels = classify_risk_levels(df)
    print("[7] Répartition des niveaux de risque ✅")
    print(risk_levels["risk_level"].value_counts(), "\n")

    # ====== Contexte LLM ======
    risk_summary = {
        "min": float(full["risk_score"].min()),
        "mean": float(full["risk_score"].mean()),
        "max": float(full["risk_score"].max()),
    }

    risk_distribution = (
        risk_levels["risk_level"]
        .value_counts(normalize=True)
        .round(3)
        .to_dict()
    )

    # Colonnes utiles pour le LLM
    cols_for_llm = [
        "machine_id", "Type", "risk_score",
        "Machine failure", "total_anomalies",
        "Tool wear [min]", "Torque [Nm]", "Rotational speed [rpm]", "Process temperature [K]",
        "torque_anomaly", "process_temp_anomaly", "rpm_anomaly", "toolwear_anomaly"
    ]
    cols_for_llm = [c for c in cols_for_llm if c in top_risk.columns]
    top_risky_df = top_risk[cols_for_llm].copy()

    # Renommer pour que le LLM comprenne
    rename_map = {
        "Type": "machine_type",
        "Machine failure": "machine_failure",
        "Tool wear [min]": "tool_wear_min",
        "Torque [Nm]": "torque_nm",
        "Rotational speed [rpm]": "rpm",
        "Process temperature [K]": "process_temp_k",
    }
    top_risky_df = top_risky_df.rename(columns=rename_map)

    # ✅ Signals robustes (pas de "is True")
    def _b(x) -> bool:
        try:
            return bool(x)
        except Exception:
            return False

    def build_signals(row):
        s = []
        if _b(row.get("toolwear_anomaly")):
            s.append("toolwear_anomaly")
        if _b(row.get("torque_anomaly")):
            s.append("torque_anomaly")
        if _b(row.get("rpm_anomaly")):
            s.append("rpm_anomaly")
        if _b(row.get("process_temp_anomaly")):
            s.append("process_temp_anomaly")
        try:
            if int(row.get("machine_failure", 0)) == 1:
                s.append("machine_failure")
        except Exception:
            pass
        return s

    top_risky_df["signals"] = top_risky_df.apply(build_signals, axis=1)

    # Garder uniquement des champs utiles (anti bruit)
    keep_cols = [
        "machine_id", "machine_type", "risk_score",
        "machine_failure", "total_anomalies",
        "tool_wear_min", "torque_nm", "rpm", "process_temp_k",
        "signals"
    ]
    keep_cols = [c for c in keep_cols if c in top_risky_df.columns]
    top_risky_df = top_risky_df[keep_cols].copy()

    # Debug utile : vérifier signals
    if VERBOSE:
        print("[DEBUG] machine_id + signals (top10):")
        print(top_risky_df[["machine_id", "signals"]].to_string(index=False), "\n")

    top_risky = top_risky_df.to_dict(orient="records")

    prompt = build_maintenance_prompt(
        kpis=kpis,
        risk_summary=risk_summary,
        top_risky=top_risky,
        risk_distribution=risk_distribution
    )

    print("\n==============================")
    print(" ANALYSE LLM (Ollama local)")
    print("==============================\n")

    analysis_text = ollama_generate(prompt, model=MODEL_NAME)
    print(analysis_text)

    print("\n✅ Fin pipeline.\n")


if __name__ == "__main__":
    main()
