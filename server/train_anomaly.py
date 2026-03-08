import sqlite3
import json
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

DB_PATH = "/var/lib/memory_service/net/net.db"

conn = sqlite3.connect(DB_PATH)

rows = conn.execute(
    "SELECT meta_json FROM net_memories"
).fetchall()

features = []

for r in rows:

    try:
        meta = json.loads(r[0])
    except:
        continue

    #ignoring non packet entries
    if "layers" not in meta:
        continue

    layers = meta["layers"]

    network = layers.get("network", {})
    transport = layers.get("transport", {})
    packet = meta.get("packet", {})

    protocol = network.get("protocol_number", 0)
    src_port = transport.get("src_port", 0)
    dst_port = transport.get("dst_port", 0)
    size = packet.get("bytes", 0)

    features.append([
        protocol,
        src_port,
        dst_port,
        size
    ])

print("Packets used for training:", len(features))

if len(features) < 50:
    raise RuntimeError("Not enough packet data for training")

X = np.array(features)

model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42
)

model.fit(X)

joblib.dump(model, "anomaly_model.pkl")

print("Anomaly model trained successfully")