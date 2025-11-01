from abc import ABC, abstractmethod
from typing import Callable

class BaseSampler(ABC):
    # given the prompt, generate samples. Can be intermediate samples as well.
    @abstractmethod
    def generate(prompt: str):
        pass

    # sample the current set of trajectories, given the reward
    @abstractmethod
    def sample(reward_fn: 'Callable[[str], float]'):
        pass

    