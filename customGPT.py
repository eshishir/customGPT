# pip install --force-reinstall typing-extensions==4.7.1 --user
# pip install --force-reinstall openai==0.28 --user

import openai
import os

# Replace 'your-api-key' with your actual OpenAI API key
openai.api_key = os.getenv('OPENAI_KEY')

class ChatGPTConversation:
    def __init__(self, model="gpt-4", temperature=0.7, max_tokens=1000, history_limit=None, initial_prompts=None):
        """
        Initialize the ChatGPT conversation.

        :param model: The GPT model to use.
        :param temperature: Sampling temperature for randomness.
        :param max_tokens: Maximum number of tokens for the response.
        :param history_limit: Number of exchanges to keep in history (None means unlimited).
        :param initial_prompts: List of initial prompts to set the context.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history_limit = history_limit
        self.conversation_history = []

        # Add initial prompts to the conversation history if provided
        if initial_prompts is not None:
            for prompt in initial_prompts:
                self.add_user_message(prompt)
    
    def add_user_message(self, prompt):
        """Add a user message to the conversation history."""
        self.conversation_history.append({"role": "user", "content": prompt})
        self._enforce_history_limit()

    def add_assistant_message(self, message):
        """Add an assistant message to the conversation history."""
        self.conversation_history.append({"role": "assistant", "content": message})
        self._enforce_history_limit()

    def _enforce_history_limit(self):
        """
        Ensure the conversation history doesn't exceed the history limit.
        If history_limit is None, do not trim the conversation history.
        """
        if self.history_limit is not None and len(self.conversation_history) > self.history_limit * 2:
            # Keep only the last 'history_limit' exchanges (each exchange includes a user and assistant message)
            self.conversation_history = self.conversation_history[-self.history_limit * 2:]

    def get_response(self, prompt):
        """Generate a response from ChatGPT based on the current conversation history."""
        self.add_user_message(prompt)
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.conversation_history,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        
        assistant_message = response.choices[0].message['content']
        self.add_assistant_message(assistant_message)
        
        return assistant_message

if __name__ == "__main__":
    # Define initial prompts to set the context
    initial_prompts = [
        "You are a helpful assistant.",
        "You always respond as a sailor."
    ]
    
    # Create a ChatGPTConversation object with the initial prompts and default (unlimited history)
    chat = ChatGPTConversation(model="gpt-4", initial_prompts=initial_prompts)

    # Example conversation loop
    while True:
        # Get user input
        prompt = input("You: ")
        if prompt.lower() in {"exit", "quit"}:
            print("Exiting the chat. Goodbye!")
            break

        # Get response from the assistant
        response = chat.get_response(prompt)
        print("ChatGPT:", response)
