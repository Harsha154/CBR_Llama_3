import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, device):
    # Load tokenizer and model from the specified local path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.to(device)
    return tokenizer, model

def generate_response(tokenizer, device, model, prompt, temperature=0.1, no_repeat_ngram_size=2):
    # Encode the prompt with the special token for the beginning of the sequence
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Enable autocast to use float16 for eligible operations
    # Generate the output
    outputs = model.generate(**inputs,
                            temperature=temperature, # reduces randomness in reponses. Model's outputs more deterministic and conservative, favoring more likely choices
                            max_new_tokens=50,
                            repetition_penalty=1.2,
                            top_p=0.95,
                            num_return_sequences=1, # number of independent sequences to generate from the same context
                            no_repeat_ngram_size=no_repeat_ngram_size, #  reducing redundancy and improving the cohesiveness of the generated text
                            eos_token_id=tokenizer.eos_token_id) # token ID used to signify the end of a sequence

    # Decode and return the model output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Specify the path to your local model directory
model_path = "/big/Meta-Llama-3-8B-Instruct"
tokenizer, model = load_model(model_path, device)

# Load the CSV file
csv_file = '/home/hs875/Llama-3/ground_truth.csv'
df = pd.read_csv(csv_file)

# Add a new column for the model's responses
df['Response from Llama'] = ''

# Iterate over the rows in the DataFrame and generate responses
for index, row in df.iterrows():
    Desc = row['Description']
    prompt = f"The following text describes an indication of a drug. What's the indication for? Each answer should be on a new line. Do not include any other information or formatting, just the names. \nText: {Desc}"
    response = generate_response(tokenizer, device, model, prompt)
    answer = response.replace(prompt, "")
    answer_split_by_note = answer.lower().split("note")
    final_answer = answer_split_by_note[0].lower()
    print(answer)
    print(final_answer)
    df.at[index, 'Instruction Prompt'] = prompt
    df.at[index, 'Original Response'] = answer
    df.at[index, 'Reponse after Note Splicing'] = final_answer
    df.to_csv('/home/hs875/Llama-3/ground_truth_edited.csv', index=False)
