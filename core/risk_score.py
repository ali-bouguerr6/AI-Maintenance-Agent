import pandas as pd

def compute_risk_score(df, anomalies):

    # normalisation usure outil (max 300 min dans dataset)
    df["toolwear_norm"] = df["Tool wear [min]"] / 300

    # risk score final
    df["risk_score"] = (
        df["toolwear_norm"] * 30
        + anomalies["torque_anomaly"] * 20
        + anomalies["process_temp_anomaly"] * 20
        + anomalies["rpm_anomaly"] * 15
        + df["Machine failure"] * 15
    )

    return df[["risk_score"]]

def classify_risk_levels(df):
    df["risk_level"] = "safe"
    df.loc[df["risk_score"] >= 30, "risk_level"] = "warning"
    df.loc[df["risk_score"] > 60, "risk_level"] = "danger"
    return df[["risk_level"]]
