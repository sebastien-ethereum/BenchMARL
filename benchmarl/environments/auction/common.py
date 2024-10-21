from typing import Callable, Dict, List, Optional

from benchmarl.environments.common import Task
from benchmarl.utils import DEVICE_TYPING

from tensordict import TensorDict
from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase
from benchmarl.environments.auction.auction_env import AuctionEnv

class AuctionTask(Task):

    FIRST_PRICE = None  # Loaded automatically from conf/task/auction/first_price
    SECOND_PRICE = None  # Loaded automatically from conf/task/auction/second_price

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        return lambda: AuctionEnv(
            n_agents=self.config.get("n_bidders", 3),
            seed=seed,
            device=device,
        )

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        return 1

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return {"agents": [f"agent_{i}" for i in range(self.config.get("n_bidders", 3))]}

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        return env.full_observation_spec

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        return env.full_action_spec

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        return env.full_state_spec

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        return None

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        return None

    @staticmethod
    def env_name() -> str:
        return "auction"

    def log_info(self, batch: TensorDict) -> Dict[str, float]:
        return {}
