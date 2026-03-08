import joblib

MODEL_PATH = "anomaly_model.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def extract_features(meta):

    network = meta["layers"]["network"]
    transport = meta["layers"]["transport"]
    packet = meta["packet"]

    protocol = network.get("protocol_number", 0)
    src_port = transport.get("src_port", 0)
    dst_port = transport.get("dst_port", 0)
    size = packet.get("bytes", 0)

    return [
        protocol,
        src_port,
        dst_port,
        size
    ]


def predict(model, features):

    score = model.decision_function([features])[0]
    pred = model.predict([features])[0]

    return {
        "anomaly": bool(pred == -1),
        "score": float(score)
    }