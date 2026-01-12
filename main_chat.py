import json

from core.loader import load_dataset
from core.kpis import compute_kpis
from core.anomalies import detect_anomalies
from core.risk_score import compute_risk_score, classify_risk_levels
from core.explain_llm import ollama_generate

from rag.query import rag_search, format_rag_context  # ✅ RAG

MODEL_NAME = "qwen2.5:1.5b"


def normalize_machine_id(raw: str) -> str:
    s = raw.strip()
    if not s:
        return s
    s = s.replace(" ", "")
    if s.startswith("MM"):
        s = "M" + s[2:]
    if not s.startswith("M"):
        s = "M" + s
    return s



def build_chat_prompt(
    question: str,
    kpis: dict,
    risk_summary: dict,
    risk_distribution: dict,
    top10: list,
    rag_context: str = "",
    rag_sources: list | None = None,
    machine_focus: dict | None = None
) -> str:
    """
    Prompt chat strict + RAG (démo wow).
    Le RAG est utilisé pour justifier les actions, pas pour identifier les machines.
    """
    ctx = {
        "question": question,
        "kpis": kpis,
        "risk_summary": risk_summary,
        "risk_distribution": risk_distribution,
        "top10": top10,
        "rag_context": rag_context,
        "rag_sources": rag_sources or []
    }
    if machine_focus is not None:
        ctx["machine_focus"] = machine_focus

    ctx_json = json.dumps(ctx, ensure_ascii=False)

    return (
        "Tu es un ingénieur maintenance industrielle.\n"
        "Tu réponds STRICTEMENT à partir du JSON.\n\n"
        "Règles:\n"
        "- N'invente pas de seuils/normes.\n"
        "- N'invente pas d'identifiants machine.\n"
        "- Si une info n'est pas dans le JSON, dis: 'info non disponible'.\n"
        "- Réponse courte et actionnable.\n\n"
        "RAG (documents internes):\n"
        "- Si rag_context contient des extraits, utilise-les UNIQUEMENT pour justifier les actions recommandées.\n"
        "- Le diagnostic des machines (quelles machines sont à risque) doit venir des données (top10/machine_focus/risk_score).\n"
        "- À la fin, ajoute une ligne: Sources: <liste> si rag_sources n'est pas vide.\n\n"
        "Réponds au format:\n"
        "1) Réponse directe (2-4 lignes)\n"
        "2) Preuves (puces courtes: champs + valeurs/signals)\n"
        "3) Actions recommandées (P1/P2/P3 si possible)\n"
        "4) Sources (si disponibles)\n\n"
        f"JSON:\n{ctx_json}"
    )


def main():
    print("\n==============================")
    print(" AI MAINTENANCE AGENT - CHAT CLI (RAG + Mémoire)")
    print("==============================\n")

    # --- Build context once ---
    df = load_dataset()
    kpis = compute_kpis(df)

    anom = detect_anomalies(df)
    risk = compute_risk_score(df, anom)

    full = df.copy()
    for c in anom.columns:
        full[c] = anom[c].values
    full["risk_score"] = risk["risk_score"].values

    if "Product ID" in full.columns:
        full["machine_id"] = full["Product ID"].astype(str).str.replace(" ", "", regex=False)
        full["machine_id"] = full["machine_id"].apply(lambda s: s if s.startswith("M") else f"M{s}")
        full["machine_id"] = full["machine_id"].apply(lambda s: "M" + s[2:] if s.startswith("MM") else s)
    else:
        full["machine_id"] = full.index.astype(str).apply(lambda x: f"M{x}")

    risk_levels = classify_risk_levels(df)

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
            if int(row.get("Machine failure", 0)) == 1:
                s.append("machine_failure")
        except Exception:
            pass
        return s

    top_df = full.sort_values("risk_score", ascending=False).head(10).copy()
    top_df["signals"] = top_df.apply(build_signals, axis=1)

    keep_cols = ["machine_id", "Type", "risk_score", "signals"]
    keep_cols = [c for c in keep_cols if c in top_df.columns]
    top10 = top_df[keep_cols].rename(columns={"Type": "machine_type"}).to_dict(orient="records")

    # ✅ Mémoire conversationnelle (session)
    session_state = {
        "last_machine_id": None,
        "last_machine_focus": None,
        "last_intent": None,         # "machine" / "global" / "top" / "free"
        "last_rag_sources": [],
        "last_question": None
    }

    print("✅ Contexte chargé. Tape :")
    print("- global | top | machine <ID> | question libre")
    print("- context (voir mémoire) | reset (reset mémoire) | exit\n")

    # --- Chat loop ---
    while True:
        user_q = input(">> ").strip()
        if not user_q:
            continue

        uq = user_q.lower()

        # --- commandes ---
        if uq in ("exit", "quit", "q"):
            print("Bye 👋")
            break

        if uq == "context":
            print("\n🧠 Mémoire session:")
            print(json.dumps(session_state, indent=2, ensure_ascii=False))
            print()
            continue

        if uq == "reset":
            session_state["last_machine_id"] = None
            session_state["last_machine_focus"] = None
            session_state["last_intent"] = None
            session_state["last_rag_sources"] = []
            session_state["last_question"] = None
            print("🧠 Mémoire réinitialisée ✅\n")
            continue

        machine_focus = None

        # --- intent & question ---
        if uq.startswith("machine "):
            mid = normalize_machine_id(user_q.split(" ", 1)[1])
            rows = full[full["machine_id"] == mid]
            if len(rows) == 0:
                print(f"❌ machine_id introuvable: {mid}\n")
                continue

            r = rows.iloc[0].to_dict()
            machine_focus = {
                "machine_id": r.get("machine_id"),
                "machine_type": r.get("Type"),
                "risk_score": float(r.get("risk_score")),
                "machine_failure": int(r.get("Machine failure", 0)),
                "tool_wear_min": float(r.get("Tool wear [min]", 0.0)),
                "torque_nm": float(r.get("Torque [Nm]", 0.0)),
                "rpm": float(r.get("Rotational speed [rpm]", 0.0)),
                "process_temp_k": float(r.get("Process temperature [K]", 0.0)),
                "signals": build_signals(r),
            }

            # ✅ update mémoire
            session_state["last_machine_id"] = mid
            session_state["last_machine_focus"] = machine_focus
            session_state["last_intent"] = "machine"

            question = f"Analyse détaillée de la machine {mid}. Explique brièvement le risque et propose des actions."

        elif uq == "global":
            session_state["last_intent"] = "global"
            question = "Donne un état global du parc (KPI + risque) et 3 priorités d'action."

        elif uq == "top":
            session_state["last_intent"] = "top"
            question = "Résume le Top 10 des machines les plus à risque et les actions prioritaires."

        else:
            # ✅ questions vagues → on réutilise la dernière machine si elle existe
            vague_markers = [
                "elle", "celle", "cette", "la machine", "pourquoi", "critique",
                "danger", "actions", "p1", "p2", "p3", "procédure", "procedure", "quoi faire"
            ]
            if session_state["last_machine_id"] and any(m in uq for m in vague_markers):
                mid = session_state["last_machine_id"]
                machine_focus = session_state["last_machine_focus"]
                question = f"Contexte: machine {mid}. Réponds à: {user_q}"
            else:
                question = user_q

        session_state["last_question"] = question

        # ✅ RAG search (k=2 pour limiter la lenteur)
        hits = rag_search(question, k=2)
        rag_context, rag_sources = format_rag_context(hits)

        session_state["last_rag_sources"] = rag_sources

        if rag_sources:
            print("📚 RAG:", ", ".join(rag_sources))
        else:
            print("📚 RAG: aucune source")

        prompt = build_chat_prompt(
            question=question,
            kpis=kpis,
            risk_summary=risk_summary,
            risk_distribution=risk_distribution,
            top10=top10,
            rag_context=rag_context,
            rag_sources=rag_sources,
            machine_focus=machine_focus
        )

        try:
            answer = ollama_generate(prompt, model=MODEL_NAME)
            print("\n" + answer + "\n")
        except Exception as e:
            print(f"⚠️ Erreur LLM: {e}\n")


if __name__ == "__main__":
    main()
