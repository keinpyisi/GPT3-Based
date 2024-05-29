
    #                   _oo0oo_
    #                  o8888888o
    #                  88" . "88
    #                  (| -_- |)
    #                  0\  =  /0
    #                ___/`---'\___
    #              .' \\|     |// '.
    #             / \\|||  :  |||// \
    #            / _||||| -:- |||||- \
    #           |   | \\\  -  /// |   |
    #           | \_|  ''\---/''  |_/ |
    #           \  .-\__  '-'  ___/-. /
    #         ___'. .'  /--.--\  `. .'___
    #      ."" '<  `.___\_<|>_/___.' >' "".
    #     | | :  `- \`.;`\ _ /`;.`/ - ` : | |
    #     \  \ `_.   \_ __\ /__ _/   .-` /  /
    # =====`-.____`.___ \_____/___.-`___.-'=====
    #                   `=---='


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from transformers import GPTNeoForCausalLM, AutoTokenizer
import random
import time

# Load model and tokenizer
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

# User input
user_input = "こんにちは"

# Encode user input
input_ids = tokenizer.encode(user_input, return_tensors='pt')

# Calculate max length based on the length of the user input
max_length = input_ids.shape[-1] + 1000  # You can adjust the additional length as needed 2048 is maximum

# Define dynamic ranges for temperature, top_k, and top_p
temperature_range = (0.5, 1.0)
top_k_range = (20, 50)
top_p_range = (0.8, 0.95)

# Generate random values within the defined ranges
temperature = random.uniform(*temperature_range)
top_k = random.randint(*top_k_range)
top_p = random.uniform(*top_p_range)

# Generate response with dynamic parameters
output = model.generate(
    input_ids, # Input token IDs
    max_length=max_length,  # Maximum length of the generated text
    pad_token_id=tokenizer.eos_token_id, # Token ID for padding
    temperature=temperature,  # Control randomness of generation
    top_k=top_k, # This parameter limits the sampling to the top k most likely next words. It filters out unlikely words and ensures that the model only considers a fixed number of tokens with the highest probabilities for each step of generation.
    top_p=top_p, # Also known as nucleus sampling or nucleus filtering, this parameter controls the cumulative probability mass considered during sampling. It restricts the sampling to the smallest set of tokens whose cumulative probability exceeds the probability top_p. In other words, it dynamically adjusts the number of tokens considered based on their probability distribution, ensuring diversity while maintaining a certain level of quality in the generated text.
    do_sample=True   # Whether to use sampling during generation
)

# Decode and print response
generated_response = tokenizer.decode(output[0], skip_special_tokens=True)
# Typing effect
for word in generated_response.split():
    for char in word:
        print(char, end='', flush=True)
        time.sleep(0.05)  # Adjust the typing speed as needed
    print(' ', end='', flush=True)
    time.sleep(0.2)  # Pause slightly between words