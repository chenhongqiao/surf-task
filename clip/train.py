from model import CLIPClassifier, clip_preprocess
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
from matplotlib import pyplot as plt
import pickle

def train(model: CLIPClassifier, train_loader: DataLoader, validation_set: tuple, lr: float, epochs: int):
    model.to("cuda")

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    val_inputs, val_labels = validation_set
    val_inputs, val_labels = val_inputs.to("cuda"), val_labels.to("cuda")

    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_inputs)
            val_loss = loss_fn(val_outputs, val_labels).item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        train_loss_history.append(avg_loss)
        val_loss_history.append(val_loss)
        
        torch.save(model.clf_head, f"./clip/checkpoints/{epoch+1}.pt")

    plt.ylim(0, 0.2)
    plt.plot(range(1, epochs+1), train_loss_history, label="Training Loss")
    plt.plot(range(1, epochs+1), val_loss_history, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.savefig("./clip/figs/learning_curve.pdf")
    
    print("Training complete!")

if __name__ == "__main__":
    with open("./data/caltech101_splits/train.pkl", "rb") as f:
        train_set = pickle.load(f)
    
    with open("./data/caltech101_splits/validation.pkl", "rb") as f:
        validation_set = pickle.load(f)
    
    train_set = clip_preprocess(train_set)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    
    validation_set = clip_preprocess(validation_set)
    val_x, val_y = zip(*validation_set)
    validation_set = (torch.stack(val_x), torch.tensor(val_y))
    
    model = CLIPClassifier()
    
    train(model, train_loader, validation_set, 1e-3, 10)
    
    