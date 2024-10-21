from typing import Optional
import torch
from tensordict import TensorDict
from torchrl.data import CompositeSpec, BoundedTensorSpec, DiscreteTensorSpec
from torchrl.envs import EnvBase
from benchmarl.utils import DEVICE_TYPING
import csv
from datetime import datetime


# Input = "State" * Action
# Output = "Observation" * Reward * Done
# TED = Input * Output
#
# N.B. "State" and "Observation" are root level
# and also egregious misnomers
# and not conflating them causes bizarre errors


class AuctionEnv(EnvBase):

    def __init__(self, n_agents: int, seed: Optional[int], device: DEVICE_TYPING):
        super().__init__(device=device)
        self.n_agents = n_agents
        self.rng = None
        self.winner = None
        self.price = None

        self.state_spec = CompositeSpec(
            agents=CompositeSpec(
                true_value=BoundedTensorSpec(
                    shape=(self.n_agents, 1),
                    low=0.0,
                    high=100.0
                ),
                shape=(self.n_agents,)
            )
        )

        self.action_spec = CompositeSpec(
            agents=CompositeSpec(
                action=BoundedTensorSpec(
                    shape=(self.n_agents, 1),
                    low=0.0,
                    high=200.0
                ),
                shape=(self.n_agents,)
            )
        )

        self.observation_spec = CompositeSpec(
            agents=CompositeSpec(
                true_value=BoundedTensorSpec(
                    shape=(self.n_agents, 1),
                    low=0.0,
                    high=100.0
                ),
                shape=(self.n_agents,)
            )
        )

        self.reward_spec = CompositeSpec(
            agents=CompositeSpec(
                reward=BoundedTensorSpec(
                    shape=(self.n_agents, 1),
                    low=-100.0,
                    high=100.0
                ),
                shape=(self.n_agents,)
            )
        )

        self.done_spec = DiscreteTensorSpec(n=2, shape=(1,), dtype=torch.bool)

        self._set_seed(seed)

        # Initialize logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'sample_log_{timestamp}.csv'
        self.file_handle = open(filename, 'a', newline='')
        self.csv_writer = csv.writer(self.file_handle)
        self.sample_log_counter = 0

    def __del__(self):
        if hasattr(self, 'file_handle') and not self.file_handle.closed:
            self.file_handle.close()

    def sample_log(self, x):
        self.sample_log_counter += 1
        if self.sample_log_counter % 1000 == 0:
            self.csv_writer.writerow(x.cpu().numpy().flatten())
            self.file_handle.flush()
            print(x)

    def _set_seed(self, seed: int):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _reset(self, _: TensorDict) -> TensorDict:
        true_values = torch.rand(self.n_agents, generator=self.rng, device=self.device) * 100

        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "true_value": true_values.unsqueeze(-1)
                    },
                    batch_size=(self.n_agents,)
                )
            }
        )

    def _step(self, td: TensorDict) -> TensorDict:
        bids = td["agents"]["action"].squeeze(-1)

        sorted_bids, sorted_indices = torch.sort(bids, descending=True)
        self.winner = sorted_indices[0]
        self.price = sorted_bids[1] if len(sorted_bids) > 1 else sorted_bids[0]

        true_values = td["agents"]["true_value"].squeeze(-1)
        rewards = torch.zeros(self.n_agents, device=self.device)
        rewards[self.winner] = true_values[self.winner] - self.price

        self.sample_log(bids - true_values)

        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "true_value": true_values.unsqueeze(-1),
                        "reward": rewards.unsqueeze(-1)
                    },
                    batch_size=(self.n_agents,)
                ),
                "done": torch.ones(1, dtype=torch.bool, device=self.device)
            }
        )
