from datasets import load_dataset
from datasets import Dataset

## kaggle emotions: angry, disgust, fear, happy, neutral, sad, surprise
emotion_map = {
    0: 'neutral',
    1: 'angry',
    2: 'disgust',
    3: 'fear',
    4: 'happy',
    5: 'sad',
    6: 'surprise'
}

### Adds context token
def build_context_response_pairs(dialogs, emotions, emotion_map=emotion_map):
    data = {
        "input": [],
        "response": []
    }

    for dialog, emo in zip(dialogs, emotions):
        for i in range(len(dialog) - 1):
            emotion_label = emotion_map[emo[i + 1]]
            context = dialog[i]
            response = dialog[i + 1]

            input_text = f"emotion: {emotion_label} | context: {context}"

            data["input"].append(input_text)
            data["response"].append(response)

    return Dataset.from_dict(data)


def get_fine_tuning_data(link='li2017dailydialog/daily_dialog',trust_remote_code = True):

    data = load_dataset(link,trust_remote_code = trust_remote_code)

    train_data = data['train']
    dialogs = train_data['dialog']
    emotions = train_data['emotion']

    train_data = build_context_response_pairs(dialogs, emotions)

    validation_data = data['validation']
    dialogs = validation_data['dialog']
    emotions = validation_data['emotion']

    validation_data = build_context_response_pairs(dialogs, emotions)

    test_data = data['test']
    dialogs = test_data['dialog']
    emotions = test_data['emotion']

    test_data = build_context_response_pairs(dialogs, emotions)

    return train_data, validation_data, test_data


if __name__ == '__main__':
    train_data, validation_data, test_data = get_fine_tuning_data()
    train_data.to_json("response_model/train_data.jsonl")
    validation_data.to_json("response_model/validation_data.jsonl")
    test_data.to_json("response_model/test_data.jsonl")