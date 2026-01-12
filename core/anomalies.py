import pandas as pd

def detect_anomalies(df):
    
    anomalies = pd.DataFrame()

    # règle couple
    anomalies["torque_anomaly"] = df["Torque [Nm]"] > 55

    # règle temperature process
    anomalies["process_temp_anomaly"] = df["Process temperature [K]"] > 315

    # règle vitesse rotation
    anomalies["rpm_anomaly"] = (df["Rotational speed [rpm]"] < 1300) | \
                               (df["Rotational speed [rpm]"] > 1700)

    # règle usure outil
    anomalies["toolwear_anomaly"] = df["Tool wear [min]"] > 200

    # calcul du total anomalies par machine
    anomalies["total_anomalies"] = anomalies.sum(axis=1)

    return anomalies
