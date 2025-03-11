import time
import torch
from typing import Optional
from pydantic import BaseModel
from abc import ABC, abstractmethod

from captionise.store import Job


class RewardEvent(BaseModel):
    """
    Contains aggregated reward info for all responses in a batch.
    Could store intermediate metrics like final WER or prompt correctness.
    """
    reward_name: str
    rewards: torch.Tensor       # final reward tensor
    batch_time: float          # how long it took to compute the batch’s rewards
    extra_info: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True


class BatchRewardOutput(BaseModel):
    """
    Output container returned by get_rewards(),
    storing the updated rewards tensor plus any additional data.
    """
    rewards: torch.Tensor
    extra_info: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True


class BatchRewardInput(BaseModel):
    """
    Input container for batch reward calculation.
    Example usage: you might store computed metrics (like WER or accuracy),
    along with a reference to the job and any extra data you need.
    """
    scores: torch.Tensor        # e.g., a tensor of text-based or STT-based metrics
    top_score: float            # if relevant, the best score in the batch
    job: Job                    # reference to the job

    class Config:
        arbitrary_types_allowed = True


class BaseReward(ABC):
    """
    Abstract base class for implementing different types of reward calculations.
    Subclasses must override name, get_rewards, and calculate_final_reward.
    """

    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of this reward strategy (e.g. 'WERReward', 'AccuracyReward').
        """
        ...

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize with any hyperparameters or references needed.
        """
        pass

    @abstractmethod
    async def get_rewards(
        self, data: BatchRewardInput, rewards: torch.Tensor
    ) -> BatchRewardOutput:
        """
        Core logic to compute partial or intermediate reward values,
        possibly based on data.scores (which might be WER or confidence).
        Should return a BatchRewardOutput containing updated rewards.
        """
        pass

    @abstractmethod
    async def calculate_final_reward(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Takes the intermediate rewards from get_rewards() and applies
        final transformations (e.g. normalization, scaling, etc.).
        """
        pass

    async def setup_rewards(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Prepare an initial rewards tensor, e.g. zeros matching scores.size(0).
        Subclasses can override for more advanced initialization if needed.
        """
        return torch.zeros(len(scores))

    async def forward(self, data: BatchRewardInput) -> RewardEvent:
        """
        High-level method that runs the entire reward pipeline:
         1) Initialize rewards via setup_rewards()
         2) Call get_rewards()
         3) Apply calculate_final_reward()
         4) Return a RewardEvent with the final rewards and timing info
        """
        # Step 1: Initialize the reward vector
        t0: float = time.time()
        self.rewards: torch.Tensor = await self.setup_rewards(scores=data.scores)

        # Step 2: Get partial or intermediate rewards
        batch_rewards_output: BatchRewardOutput = await self.get_rewards(
            data=data, rewards=self.rewards
        )

        # Step 3: Final transformation
        batch_rewards_output.rewards = await self.calculate_final_reward(
            rewards=batch_rewards_output.rewards
        )

        # Step 4: Compute time
        batch_rewards_time: float = time.time() - t0

        return RewardEvent(
            reward_name=self.name(),
            rewards=batch_rewards_output.rewards,
            batch_time=batch_rewards_time,
            extra_info=batch_rewards_output.extra_info,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
