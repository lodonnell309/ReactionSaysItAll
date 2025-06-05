# ReactionSaysItAll

**ReactionSaysItAll** is a CS 7643 Deep Learning project that explores how facial expressions can drive natural language responses. The app uses a Convolutional Neural Network (CNN) to analyze a user's facial reaction to a text message and then generates a response based on that reaction. We fine-tuned Google’s T5-small model for text generation, with optional support for integrating Gemini for more advanced responses. The final project paper can be found in project deliverables. 

---

## ⚙️ Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/lodonnell309/ReactionSaysItAll.git
   cd ReactionSaysItAll

2. **Install Requirements**
   ```bash 
   pip install -r requirements.txt

3. **Run the app**
   ```bash
   python app.py
   # or, depending on your environment
   python3 app.py
   
## 🔐 Using Gemini (Optional)
To enable Gemini language model integration:

1. Create a file named api_key.py in the root directory.

2. Add your API key in the following format:
   ```python
   api_key = "your_gemini_api_key_here"
