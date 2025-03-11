from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseEvaluator(ABC):
    """
    Abstract base evaluator class for the Captionise project.
    A typical usage might be measuring a metric (e.g., Word Error Rate,
    textual correctness, or any other domain-specific measure).
    """

    @abstractmethod
    def name(self) -> str:
        """
        Return a name or identifier for this evaluator, e.g. 'WER', 'AccuracyEvaluator', etc.
        """
        ...

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize your evaluator with any configuration or hyperparameters.
        """
        pass

    @abstractmethod
    def _evaluate(self, data: Dict[str, Any]) -> float:
        """
        Core logic for calculating the evaluation metric.
        
        Args:
            data (Dict[str, Any]): A dictionary of relevant data
              - might include reference text, predicted text, timestamps, etc.

        Returns:
            float: The computed metric or score.
        """
        pass

    @abstractmethod
    def _validate(self):
        """
        Optionally perform checks to ensure the data is valid for evaluation,
        or to confirm the evaluator is ready to run. This is a placeholder
        you can adapt for your domain-specific logic.
        """
        pass

    def forward(self, data: Dict[str, Any]) -> Any:
        """
        A public method that calls the internal _evaluate logic.
        Subclasses might override or enhance this method if needed.
        """
        return self._evaluate(data=data)

    def __repr__(self) -> str:
        """
        String representation, e.g. "WER()" or "CaptionEvaluator()".
        """
        return f"{self.__class__.__name__}<{self.name()}>"
