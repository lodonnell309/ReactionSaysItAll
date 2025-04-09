from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

model_path = "lodonnell32/ReactionSaysItAll"

tokenizer = T5Tokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=50).to(device)
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    print(generate_response("emotion: fear | How are you feeling today?"))
    print(generate_response("emotion: sad | context: How is your day going?"))