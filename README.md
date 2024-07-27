# TravelCompanion.ai

TravelCompanion.ai is a Streamlit application that leverages advanced AI to provide insights and recommendations based on attractions around the world. Simply enter a location to discover key attractions and receive personalized insights to enhance your travel experience.

## Features

- Get personalized travel insights based on the location you provide.
- Ask follow-up questions to gain deeper insights into the attractions.
- Automatically scrolls to the latest response for a seamless user experience.
- Utilizes OpenAI for generating insights and Pinecone for efficient vector storage and retrieval.
- Measures **Latency** of each interaction, providing insights into the performance and speed of the AI responses.
- Allows users to **Rate the Faithfulness** of the AI-generated answers, ensuring accuracy and reliability of the information provided.

## Getting Started

### Prerequisites

- Python 3.7 or later
- Streamlit
- OpenAI API Key
- Pinecone API Key

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/HSodhani/TravelCompanion.ai.git
    cd TravelCompanion.ai
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  
    
    # On Windows use 
    `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Set your API keys in the `.streamlit/secrets.toml` file:
    ```toml
    [secrets]
    PINECONE_API_KEY = "your_pinecone_api_key"
    OPENAI_API_KEY = "your_openai_api_key"
    ```

5. Prepare your data file (`data.json`) and place it in the appropriate directory as specified in the script:
    ```json
    [
        {
            "location": "Berlin, Germany",
            "attractions": ["Brandenburg Gate", "Berlin Wall Memorial", ...]
        },
        ...
    ]
    ```

### Running the Application

To run the application locally, execute the following command:
```bash
streamlit run main.py
```

### Youtube Video Link:
https://youtu.be/pCz2vvi7BWs

### Acknowledgements
- OpenAI for providing the language model.
- Pinecone for vector storage and retrieval.
- Streamlit for the interactive web app framework.

### Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.
