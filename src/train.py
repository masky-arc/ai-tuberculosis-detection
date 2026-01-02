from data_loader import load_datasets
from model import build_tb_model

def train_model(data_dir: str, epochs=10):
    train_data, val_data = load_datasets(data_dir)

    model = build_tb_model()

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs
    )

    model.save("models/trained_model.h5")
    return history

if __name__ == "__main__":
    train_model("data/processed")
