# CBR_Llama_3
This repository contains the source data and scripts used for the Case Base Reasoning Research.

Source Data:
- The primary indications used for the following scripts are from the text corpus file named _indications-rx-.txt_ (located in one drive folder cbr-llm).
- To run experiments, we randomly selected 120 indication excerpts from the corpus and saved them in the file named _120_random.csv_.
- The ground truth (right indication determined by Gwenlyn) for these 120 texts is located in the _gwenlyn_ground_truth_repo.xlsx_ and will be used as the benchmark to compare the performance of our experiments.

Model:
- These scripts use a downloaded Llama-3-8b-Instruct model from hugging face: [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
- Since the language model is downloaded to a common location, you can run this script with the same model location. Change the input file directory as needed.


To experiment without the use of cases:
- To use the llama-3 model without case-based reasoning, you can run the script _llama3_no_cbr.py_. 
- The script iterates over the 120 randomly selected indications from the repo with the initial prompt and submits the prompt to the Llama-3-8b model.
- The script also implements a post-processing step to splice notes after the responses.
- The results of this experiment are saved as a CSV file in a given location with the following information. 1. Indication Excerpt 2. Ground Truth, 3. Prompt submitted to the model, 4. Original Response from Llama-3, 5. Response after note splicing post-processing step.

To experiment with the use of cases:
- To conduct a similarity assessment with the existing cases and submit the appropriate prompt to llama-3, you can run the _generate_using_cases.py_ script.
- The script uses the existing cases in _updated_cases.json_ file and runs a similarity assessment to determine the most similar case text and uses the appropriate prompt for that case.
- The script iterates over the 120 randomly selected indications from the repo with the case prompt and submits the prompt to the Llama-3-8b model.
- This script also implements a post-processing step to splice notes after the responses.
- The results of this experiment are saved as a CSV file in a given location with the following information. 1. Indication Excerpt 2. Ground Truth, 3. Prompt submitted to the model, 4. Original Response from Llama-3, 5. Response after note splicing post-processing step, 6. Case ID (ID of the case the text was most similar to), 7. Solution of the case (Prompt used for the case)
  
