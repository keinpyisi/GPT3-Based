from transformers import GPTNeoForCausalLM, AutoTokenizer
from torch.optim import AdamW
import torch
from torch.utils.data import Dataset, DataLoader

# Define your conversational dataset
class ConversationDataset(Dataset):
    def __init__(self, tokenizer, conversations, max_length):
        self.tokenizer = tokenizer
        self.conversations = conversations
        self.max_length = max_length
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        input_text = conversation["input_text"]
        target_text = conversation["target_text"]
        
        # Tokenize input and target text
        input_ids = self.tokenizer.encode(input_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        target_ids = self.tokenizer.encode(target_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        
        return input_ids.squeeze(0), target_ids.squeeze(0)

# Define max sequence length
max_length = 128

# Load pre-trained model and tokenizer
model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Setting pad_token to be EOS token
model = GPTNeoForCausalLM.from_pretrained(model_name)

# Define your fine-tuning parameters
batch_size = 4
num_epochs = 3
learning_rate = 5e-5

# Prepare your dataset (not shown here)
# Sample data for training dataset
train_data = [
    {"input_text": "How are you?", "target_text": "I'm good, thank you!"},
    {"input_text": "What's your favorite color?", "target_text": "I like blue."},
    {"input_text": "Tell me about your hobbies.", "target_text": "I enjoy reading and hiking."},
    # Add more conversations as needed
]

# Create your ConversationDataset
train_dataset = ConversationDataset(tokenizer, train_data,max_length)
# Create DataLoader for training
# Create DataLoader for training with num_workers=0
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Fine-tune the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs, labels=labels)
        loss = criterion(outputs.logits.view(-1, outputs.logits.shape[-1]), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_gpt_neo_2_7B")
tokenizer.save_pretrained("fine_tuned_gpt_neo_2_7B")
