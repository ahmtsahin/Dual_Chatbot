# Football Debate Chatbot

This project is a Streamlit application that simulates a conversation between a Manchester United fan and an Arsenal fan. The chatbot uses LangChain, Hugging Face models, and FAISS for conversational retrieval and generation.

## Features

- **Conversational Agents**: Two distinct chatbots representing Manchester United and Arsenal fans, each equipped with their own memory and facts database.
- **Customizable Prompt Templates**: Each chatbot uses a tailored prompt to generate responses that are witty and relevant.
- **Real-time Chat Simulation**: The application runs a 2-minute conversation where the chatbots debate on which football team is superior.

## Setup

### Prerequisites

- Python 3.8+
- Hugging Face API Key
- Required Python packages listed in `requirements.txt`

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/ahmtsahin/Dual_Chatbot.git
    cd Dual_Chatbot
    ```

2. **Create a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up your API keys:**

    - Create a `.env` file in the project root.
    - Add your Hugging Face API key to the `.env` file:

      ```bash
      MY_API_KEY=your_hugging_face_api_key_here
      ```

5. **Download Embedding Models and Create FAISS Indexes:**

   - Ensure that you have the necessary embeddings stored in `data/manu` and `data/arsenal` folders.
   - If the FAISS indexes are not available, you will need to generate them using your dataset.

### Running the Application

1. **Run the Streamlit app:**

    ```bash
    streamlit run dualchatbot.py
    ```

2. **Interact with the chatbots:**

   - The application will start in your default web browser.
   - The conversation will simulate a 2-minute debate between the Manchester United and Arsenal fan bots.

### Key Files

- **`dualchatbot.py`**: Main application file containing the Streamlit code and chatbot logic.
- **`key.env`**: Environment file storing API keys.
- **`data/`**: Directory containing embedding models and FAISS indexes.
- **`requirements.txt`**: List of Python dependencies.

### Customization

- **Change Chatbot Behavior:**
  - Modify the prompt templates in the `dualchatbot.py` file to adjust how each chatbot responds.
  
- **Update Embeddings:**
  - You can replace the `sentence-transformers/all-MiniLM-L6-v2` model with another model compatible with FAISS indexing.

### Future Enhancements

- Add support for more teams and fan personas.
- Implement a more sophisticated relevance-checking mechanism.
- Provide a way to dynamically adjust the length of the conversation.


## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [Hugging Face](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss)
