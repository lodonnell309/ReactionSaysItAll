from google import genai
from api_key import api_key

api_key = api_key
client = genai.Client(api_key=api_key)


def get_gemini_response(emotion='Neutral',context='How is your day going?'):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"Respond to this text message {context} given emotion {emotion}. Do not provide any additional text besides the response."
    )
    return response.text

if __name__ == '__main__':
    print(get_gemini_response(emotion='Happy',context='What are you going to do today?'))