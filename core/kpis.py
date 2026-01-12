import pandas as pd

def compute_kpis(df):

    # KPIs structurés en catégories
    kpis = {
        "production_overview": {
            "nb_samples": int(len(df)),
            "type_distribution": df["Type"]
                .value_counts(normalize=True)
                .round(2)
                .astype(float)
                .to_dict(),
        },

        "performance": {
            "air_temperature_mean_K": float(df["Air temperature [K]"].mean().round(2)),
            "process_temperature_mean_K": float(df["Process temperature [K]"].mean().round(2)),
            "torque_mean_Nm": float(df["Torque [Nm]"].mean().round(2)),
            "rpm_mean": float(df["Rotational speed [rpm]"].mean().round(2)),
            "tool_wear_mean_min": float(df["Tool wear [min]"].mean().round(2)),
        },

        "maintenance": {
            "failure_rate_percent": float(df["Machine failure"].mean().round(4) * 100),
        }
    }

    return kpis

