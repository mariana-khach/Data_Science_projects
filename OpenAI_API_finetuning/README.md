I have used Fine-tuned model "gpt-3.5-turbo-0125" for email classsification as spam and not spam using the OpenAI API (see LLM_fine_tuning_spamemail.ipynb) and serving with
FastAPI (see main.py). The fine tuning is needed for adapting a general-purpose language model to your specific needs.
I have stored training and validation data in training_dataset.jsonl and validation_dataset.jsonl files, where each line is a valid JSON object.
I have then uploaded the files to OpenAI servers using OpenAI CLI and have fine tuned the model.
