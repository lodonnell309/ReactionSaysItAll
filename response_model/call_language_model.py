from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import os
import random
import json


model_path = "lodonnell32/ReactionSaysItAll"

tokenizer = T5Tokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=50).to(device)
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_random_msg():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "test_data.jsonl")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} not found")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    random_index = random.choice(lines).strip()
    entry = json.loads(random_index)
    input_text = entry["input"]
    context = input_text.split("context:")[1].strip() if "context:" in input_text else ""
    return context


# def get_random_msg():
#     with open("test_data.jsonl", "r", encoding="utf-8") as f:
#         lines = f.readlines()
#
#     random_index = random.choice(lines).strip()
#
#     entry = json.loads(random_index)
#
#     input_text = entry["input"]
#     context = input_text.split("context:")[1].strip() if "context:" in input_text else ""
#
#     return context


def query_model(emotion='neutral',text_message='random'):
    """
    :param emotion:
    Emotions include
        - angry
        - disgust
        - fear
        - happy
        - neutral
        - sad
        - surprise
    :param text_message:
    :return: A thread of texts, if text_message is set to 'random' a random message will be "received" and be used
    to generate the response
    """
    if str.lower(text_message) == 'random':
        text_message = get_random_msg()

    full_query = f'emotion: {emotion} | {text_message}'
    return generate_response(full_query)


if __name__ == "__main__":
    print(generate_response("emotion: fear | How are you feeling today?"))
    print(generate_response("emotion: sad | context: How is your day going?"))
    print(generate_response("emotion: happy | context: Hi, my name is Liam"))
    print('---------------------')

    query_model(emotion='angry',text_message='random')