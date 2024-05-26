import pandas as pd
import json
import spacy
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def similarity_assessment(tokenizer, model, cases_data, input_text, length_sim_weight = 0.35, sent_sim_weight = 0.65):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(input_text)
    input_sentences = [sent.text for sent in doc.sents] # list of sentences in the input text
    input_text_length = len(doc) # store the length of the input text by tokens

    # intiate variables to store most similar case information
    most_sim_case_score = 0 # score to keep track of most similar case, starts with 0 (meaning no similarity)
    most_sim_case_id = None # id of the most sim case base

    for case_dict in cases_data: # iterate through each case base
        case_id = case_dict["id"]
        max_sen_similarity_score = 0 # score to keep track of most similar sentence, starts with 0 (meaning no similarity)
        case_passage_len = case_dict["Problem"]["LENGTH"] # len of the case base text

        # iterate through each sentence in the case base text    
        for case_sen in case_dict["Problem"]["SENT_TEXT"]:
            # turn each sentence of the case base text into an embedding
            case_sen_input_ids = tokenizer.encode(case_sen, return_tensors="pt").to(device)
            with torch.no_grad():
                case_sen_model_output = model(case_sen_input_ids, output_hidden_states=True)
                case_sen_input_embedding = case_sen_model_output.hidden_states[-1].mean(dim=1)

            # iterate through each sentence in the input text
            for input_sen in input_sentences:
                # turn each sentence of the input text into an embedding
                input_sen_input_ids = tokenizer.encode(input_sen, return_tensors="pt").to(device)
                with torch.no_grad():
                    input_sen_model_output = model(input_sen_input_ids, output_hidden_states=True)
                    input_sen_input_embedding = input_sen_model_output.hidden_states[-1].mean(dim=1)

                local_cosine_sim_tensor = F.cosine_similarity(input_sen_input_embedding, case_sen_input_embedding, dim=1) # find cosine similarity using case base sentence and input text sentence
                local_cosine_sim_score = local_cosine_sim_tensor.item()

                if local_cosine_sim_score > max_sen_similarity_score:
                    max_sen_similarity_score = local_cosine_sim_score # score of the most similar sentence between all input text sentence and case base text sentences

        # Compute local_length similarity between the input_text_length and case_passage_len
        local_length_sim = min(case_passage_len, input_text_length) / max(case_passage_len, input_text_length)

        # Calculate weighted vector using weight parameters
        local_weighted_vector = (local_length_sim * length_sim_weight) + (max_sen_similarity_score * sent_sim_weight)
        print(f"Case {case_id} with similarity {local_weighted_vector}")

        if local_weighted_vector > most_sim_case_score: # compare the weighted score of each case and store the most similar case
            most_sim_case_score = local_weighted_vector
            most_sim_case_id = case_id
            most_sim_case_solution = case_dict["Solution"]["Instruction_Prompt"]

    return most_sim_case_id, most_sim_case_solution


# Specify the path to your local model directory
model_path = "/big/Meta-Llama-3-8B-Instruct"
tokenizer, model = load_model(model_path, device)

# Load the CSV file
csv_file = '/home/hs875/Llama-3/ground_truth.csv'
df = pd.read_csv(csv_file)

# Get list of cases from json file
cases_json = "/home/hs875/Llama-3/updated_cases.json"
cases_data = json.load(open(cases_json))

# Iterate over the rows in the DataFrame and generate responses
for index, row in df.iterrows():
    input_text = row['Description']
    case_id, case_soln = similarity_assessment(tokenizer, model, cases_data, input_text)
    prompt = f"{case_soln}{input_text}"
    print(prompt)
    response = generate_response(tokenizer, device, model, prompt)
    answer = response.replace(prompt, "")
    answer_split_by_note = answer.lower().split("note")
    final_answer = answer_split_by_note[0].lower()
    print(answer)
    print(final_answer)
    df.at[index, 'Instruction Prompt'] = prompt
    df.at[index, 'Original Response'] = answer
    df.at[index, 'Reponse after Note Splicing'] = final_answer
    df.at[index, 'Case ID'] = case_id
    df.at[index, 'Case Solution'] = case_soln
    df.to_csv('/home/hs875/Llama-3/Second_Round_with_8_cases_97_Prompts.csv', index=False)