from abc import ABC, abstractmethod
from typing import Iterator


class LLMClient(ABC):
    """
    Abstract interface for LLM providers.

    Implementations must provide synchronous and streaming generation methods.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Get the name of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a full response for the given prompt.

        Returns:
            The complete model output as a string.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_stream(self, prompt: str) -> Iterator[str]:
        """
        Stream partial tokens/chunks for the given prompt.

        Yields:
            Incremental pieces of the model output.
        """
        raise NotImplementedError

