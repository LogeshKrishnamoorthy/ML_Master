import torch

def save_checkpoint(epoch, model, optimizer, loss, filepath):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved → {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath,map_location= "cuda" if torch.cuda.is_available() else "cpu"
)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    print(f"Checkpoint loaded → Epoch {epoch}")

    return epoch, loss
