from transformers import GPTNeoForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned model and tokenizer
model_path = "fine_tuned_gpt_neo_2_7B"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

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
    print("ChatGPT neo 2.7B:", response)
