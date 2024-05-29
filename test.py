from transformers import GPTNeoForCausalLM, AutoTokenizer
import torch
import time

# Load the fine-tuned model and tokenizer
# Load model and tokenizer
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a function to generate responses
def generate_response(prompt, max_length=100, temperature=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, 
                            max_length=max_length, 
                            temperature=temperature,
                            do_sample=True,  # Set do_sample to True for sample-based generation
                            pad_token_id=tokenizer.eos_token_id)  # Set pad_token_id for reliable results
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Example usage
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Chat ended.")
        break
    response = generate_response(user_input)
    # Typing effect
    for word in response.split():
        for char in word:
            print(char, end='', flush=True)
            time.sleep(0.05)  # Adjust the typing speed as needed
        print(' ', end='', flush=True)
        time.sleep(0.2)  # Pause slightly between words
