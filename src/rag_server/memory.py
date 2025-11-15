
from collections import deque
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure you set your OpenAI API key in the environment variables


class SimpleMemory:
    """Stores conversational messages for contextual response generation."""
    def __init__(self, max_memory_size=10):
        """Initializes the memory buffer."""
        self.memory = deque(maxlen=max_memory_size)

    def add_message(self, role, content):
        """Adds a message to the memory buffer.

        Args:
            role (str): Role of the message sender (e.g., 'user', 'assistant').
            content (str): Message content.
        """
        self.memory.append({"role": role, "content": content})

    def get_context(self):
        """Returns all stored messages as context.

        Returns:
            List[dict]: List of message dictionaries.
        """
        return list(self.memory)


class SummarizedMemory:
    """Maintains a limited memory buffer and generates a summary using a language model."""
    def __init__(self, max_memory_size=10, summary_prompt="Summarize this conversation."):
        """Initializes the memory buffer and summary prompt.

        Args:
            max_memory_size (int): Maximum number of messages to retain in memory.
            summary_prompt (str): Prompt used to guide the summary generation.
        """
        self.memory = deque(maxlen=max_memory_size)
        self.summary_prompt = summary_prompt

    def add_message(self, role, content):
        """Adds a message to the memory buffer.

        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (str): The content of the message.
        """
        self.memory.append({"role": role, "content": content})

    def get_context(self):
        """Generates a summarized version of the current memory buffer.

        Returns:
            List[dict]: A single dictionary containing the summary with a system role.
        """
        conversation = "\n".join(f"{m['role']}: {m['content']}" for m in self.memory)
        prompt = f"{self.summary_prompt}\n\n{conversation}\n\nSummary:"

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You summarize conversations."},
                {"role": "user", "content": prompt}
            ]
        )
        summary = response.choices[0].message.content.strip()
        return [{"role": "system", "content": f"Summary: {summary}"}]
