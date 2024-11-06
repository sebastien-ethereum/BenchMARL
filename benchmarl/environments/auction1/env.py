from typing import Optional
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensordict import TensorDict
from torchrl.data import Composite, Bounded, Binary
from torchrl.envs import EnvBase
from benchmarl.utils import DEVICE_TYPING


# Input = "State" * Action
# Output = "Observation" * Reward * Done
# TED = Input * Output
#
# N.B. "State" and "Observation" are root level and misnomers


class Auction1Env(EnvBase):
    """
    Environment simulating sealed-bid auctions with learning agents and NPCs.
    NPCs bid a fixed fraction of their true value.
    """

    def __init__(
        self,
        num_envs: int,
        seed: Optional[int],
        device: DEVICE_TYPING,
        price_index: int = 0,
        num_agents: int = 2,
        npcs: dict[str, float] = {"A": 0.5, "B": 0.7, "C": 0.9}
    ):
        """
        Args:
            num_envs: Number of vectorized environments
            seed: Random seed
            device: Torch device
            price_index: Index of winning price (0 = first price auction)
            num_agents: Number of learning agents
            npcs: Dictionary mapping NPC names to their bid fractions
        """
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if price_index < 0:
            raise ValueError("price_index must be non-negative")
        if num_agents < 1:
            raise ValueError("num_agents must be at least 1")
        if not npcs:
            raise ValueError("npcs dictionary cannot be empty")
        if not all(0 <= f <= 1 for f in npcs.values()):
            raise ValueError("NPC bid fractions must be between 0 and 1")

        super().__init__(device=device, batch_size=(num_envs,))

        self._set_seed(seed)

        self.num_envs = num_envs
        self.price_index = price_index
        self.num_agents = num_agents

        self.num_npcs = len(npcs)
        self.npc_names = list(npcs.keys())
        self.npc_fractions = torch.tensor([npcs[name] for name in self.npc_names], device=device).view(1, -1, 1)  # Shape: [1, num_npcs, 1]

        total_bidders = self.num_agents + self.num_npcs
        if price_index >= total_bidders:
            raise ValueError(f"price_index must be less than total number of bidders ({total_bidders})")

        self.group_map = {f"agent_{i}": [f"agent_{i}"] for i in range(self.num_agents)}
        self.npc_group_map = {f"npc_{name}": [f"npc_{name}"] for name in self.npc_names}

        self._setup_specs()

        plt.ioff()
        self.fig = None
        self.ax = None

    def _setup_specs(self):
        # Note: BenchMARL expects the _unbatched_ specs, although this is documented nowhere

        # STATE
        self.unbatched_state_spec = None

        # ACTION
        # npc_* fields are hidden from BenchMARL in Task
        # (unused here)
        self.unbatched_action_spec = Composite(
            shape=(),
            **{
                f"agent_{i}": Composite(
                    shape=(1,),
                    action=Bounded(shape=(1, 1), low=0.0, high=2.0)
                ) for i in range(self.num_agents)
            }
        )

        # OBSERVATION
        # npc_* fields are hidden from BenchMARL in Task
        # so they're not actually observed by agents (TODO: confirm this)
        # NPC rewards are included here because for some reason Task
        # doesn't seem to have a reward_spec method where they could be hidden
        # (again "Observation" is a misnomer)
        self.unbatched_observation_spec = Composite(
            shape=(),
            **{
                **{
                    f"agent_{i}": Composite(
                        shape=(1,),
                        observation=Bounded(shape=(1, 1), low=0.0, high=1.0)
                    ) for i in range(self.num_agents)
                },
                **{
                    f"npc_{name}": Composite(
                        shape=(1,),
                        reward=Bounded(shape=(1, 1), low=-2.0, high=1.0)
                    ) for name in self.npc_names
                }
            }
        )

        # REWARD
        self.unbatched_reward_spec = Composite(
            shape=(),
            **{
                f"agent_{i}": Composite(
                    shape=(1,),
                    reward=Bounded(shape=(1, 1), low=-2.0, high=1.0)
                ) for i in range(self.num_agents)
            }
        )

        # DONE
        self.unbatched_done_spec = Binary(shape=(1,), dtype=torch.bool)

        # Vectorized
        self.state_spec = None
        self.action_spec = self.unbatched_action_spec.expand(
            *self.batch_size, *self.unbatched_action_spec.shape
        )
        self.observation_spec = self.unbatched_observation_spec.expand(
            *self.batch_size, *self.unbatched_observation_spec.shape
        )
        self.reward_spec = self.unbatched_reward_spec.expand(
            *self.batch_size, *self.unbatched_reward_spec.shape
        )
        self.done_spec = self.unbatched_done_spec.expand(
            *self.batch_size, *self.unbatched_done_spec.shape
        )

    def _set_seed(self, seed: Optional[int]):
        seed = 0 if seed is None else seed
        self.rng = torch.Generator(device=self.device).manual_seed(seed)

    def _reset(self, _: TensorDict) -> TensorDict:
        # Shape: [num_envs, num_agents, 1]
        self.agent_values = torch.rand(self.num_envs, self.num_agents, 1, generator=self.rng, device=self.device)

        # Shape: [num_envs, num_npcs, 1]
        self.npc_values = torch.rand(self.num_envs, self.num_npcs, 1, generator=self.rng, device=self.device)

        return TensorDict(
            {
                **{
                    f"agent_{i}": TensorDict(
                        {
                            # Shape: [num_envs, 1, 1]
                            "observation": self.agent_values[:, i:i+1, :]
                        },
                        batch_size=(self.num_envs, 1)
                    ) for i in range(self.num_agents)
                },
                **{
                    f"npc_{name}": TensorDict(
                        {
                            # Shape: [num_envs, 1, 1]
                            "reward": torch.zeros(self.num_envs, 1, 1, device=self.device)
                        },
                        batch_size=(self.num_envs, 1)
                    ) for name in self.npc_names
                }
            },
            batch_size=(self.num_envs,)
        )

    def _step(self, td: TensorDict) -> TensorDict:
        # Shape: [num_envs, num_agents, 1]
        agent_bids = torch.cat([td[f"agent_{i}"]["action"] for i in range(self.num_agents)], dim=1)

        # Shape: [num_envs, num_npcs, 1]
        npc_bids = self.npc_values * self.npc_fractions

        # Shape: [num_envs, num_agents + num_npcs, 1]
        values = torch.cat([self.agent_values, self.npc_values], dim=1)
        bids = torch.cat([agent_bids, npc_bids], dim=1)

        # Shape: [num_envs, num_agents + num_npcs, 1]
        sorted_bids, sorted_indices = torch.sort(bids, dim=1, descending=True)

        # Shape: [num_envs, 1, 1]
        winners = sorted_indices[:, [0], :]
        winning_prices = sorted_bids[:, [self.price_index], :]
        winners_values = values.gather(1, winners)
        winners_rewards = winners_values - winning_prices

        # Shape: [num_envs, num_agents + num_npcs, 1]
        rewards = torch.zeros_like(values)
        rewards.scatter_add_(1, winners, winners_rewards)

        # Shape: [num_envs, num_agents + num_npcs, 1]
        self.render_values = values.cpu().numpy()
        self.render_bids = bids.cpu().numpy()
        self.render_rewards = rewards.cpu().numpy()

        return TensorDict(
            {
                **{
                    f"agent_{i}": TensorDict(
                        {
                            # Shape: [num_envs, 1, 1]
                            "observation": td[f"agent_{i}"]["observation"],
                            "reward": rewards[:, i:i+1, :]
                        },
                        batch_size=(self.num_envs, 1)
                    ) for i in range(self.num_agents)
                },
                **{
                    f"npc_{name}": TensorDict(
                        {
                            # Shape: [num_envs, 1, 1]
                            "reward": rewards[:, self.num_agents + i:self.num_agents + i + 1, :]
                        },
                        batch_size=(self.num_envs, 1)
                    ) for i, name in enumerate(self.npc_names)
                },
                # Shape: [num_envs, 1]
                "done": torch.ones(self.num_envs, 1, dtype=torch.bool, device=self.device)
            },
            batch_size=(self.num_envs,)
        )

    def render(self, mode="rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Only rgb_array rendering mode is supported")

        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
        else:
            self.ax.clear()

        for i in range(self.num_agents):
            values = self.render_values[:, i, 0].flatten()
            bids = self.render_bids[:, i, 0].flatten()

            sort_idx = np.argsort(values)
            values_sorted = values[sort_idx]
            bids_sorted = bids[sort_idx]

            self.ax.plot(values_sorted, bids_sorted, '-o', label=f'Agent {i}',
                        alpha=0.7, markersize=4)

        self.ax.set_xlabel('Value')
        self.ax.set_ylabel('Bid')
        self.ax.set_title('Agent Bidding Functions')
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5, label='y=x')

        plt.tight_layout()

        self.fig.canvas.draw()
        data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        return data

    def close(self) -> None:
        """Clean up matplotlib resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        super().close()
