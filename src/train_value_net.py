import torch
import config
import os
from dataset_hdc import create_dataloader
from value_net import HDCValueNet, train_value_net

def main():
    print("🧠 Starting HDC Value Net Training...")
    
    if not os.path.exists(config.HDC_DATA_FILE):
        print(f"⚠️ No HDC training data found at {config.HDC_DATA_FILE}. Skipping.")
        return

    # Check file size
    with open(config.HDC_DATA_FILE) as f:
        count = sum(1 for _ in f)
    if count < 10:
        print(f"⚠️ Not enough data points ({count}) to train. Need at least 10.")
        return

    # Create Dataloader
    dataloader = create_dataloader(config.HDC_DATA_FILE, batch_size=32)
    if not dataloader:
        print("❌ Failed to create dataloader.")
        return
        
    # Init Model
    model = HDCValueNet()
    if os.path.exists(config.VALUE_NET_PATH):
        print(f"   🔄 Loading existing weights from {config.VALUE_NET_PATH}")
        try:
            model.load_state_dict(torch.load(config.VALUE_NET_PATH))
        except:
            print("   ⚠️ Failed to load weights, starting fresh.")
    
    # Train
    print(f"   🏃 Training on {count} samples...")
    model = train_value_net(model, dataloader, epochs=5)
    
    # Save
    torch.save(model.state_dict(), config.VALUE_NET_PATH)
    print(f"✅ Model saved to {config.VALUE_NET_PATH}")

if __name__ == "__main__":
    main()
