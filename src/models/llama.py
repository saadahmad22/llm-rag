from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages.utils import get_buffer_string
from langchain_core.language_models import LLM
from llama_cpp import Llama # https://llama-cpp-python.readthedocs.io/en/latest/api-reference/

class LlamaModel:
    """A class to represent a Llama model for generating responses."""

    def __init__(self, model_path: str) -> None:
        """Initializes the Llama model with the given model path.
        
        Args:
            model_path (str): The path to the Llama model file.
        """

        self.model_path: str = model_path
        self.model: LLM = self.load_model()
        self.chat_histories = {}

    def load_model(self) -> LLM:
        """Loads the Llama model from the specified path.
        Returns:
            LLM: The loaded Llama model.
        """

        model = Llama(
            model_path=self.model_path,
            n_ctx=2 ** 16, # 2^16 is 65536 tokens
            n_batch=2 ** 10, # batching size of 1024 tokens to the model
            verbose=False
            )
        return model
    
    def get_user_history(self, user_id: str) -> InMemoryChatMessageHistory:
        """Retrieves the chat history for a given user ID. Creates a new history if it does not exist.
        Args:
            user_id (str): The ID of the user whose chat history is to be retrieved.
        Returns:
            InMemoryChatMessageHistory: The chat history for the specified user.
        """

        if user_id not in self.chat_histories:
            self.chat_histories[user_id] = InMemoryChatMessageHistory()
        return self.chat_histories[user_id]

    def generate_response(self, input_data: str, history: InMemoryChatMessageHistory) -> str:
        """Generates a response from the Llama model based on the input data and chat history.
        Args:
            input_data (str): The input data to generate a response for.
            history (InMemoryChatMessageHistory): The chat history for the user.
        Returns:
            str: The generated response from the Llama model.
        """

        input_text: str = self.preprocess_input(input_data)
        history.add_user_message(input_text)
        response: str = self.model(prompt=get_buffer_string(history.messages), max_tokens=500)
        print(response)
        history.add_ai_message(response['choices'][0]['text'])
        return self.postprocess_output(response['choices'][0]['text'])

    def preprocess_input(self, input_data: str) -> str:
        
        # Preprocess the input data before passing it to the model
        return input_data.strip()

    def postprocess_output(self, output_data: str) -> str:
        # Postprocess the output data from the model
        return output_data.strip()