from typing import Callable, Dict, List, Optional

from benchmarl.environments.common import Task
from benchmarl.utils import DEVICE_TYPING

from tensordict import TensorDictBase
from torchrl.data import Composite
from torchrl.envs import EnvBase
from benchmarl.environments.auction1.env import Auction1Env


# Note: BenchMARL expects the _unbatched_ specs, although this is documented nowhere


class Auction1Task(Task):

    AUCTION1 = None

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        return lambda: Auction1Env(
            num_envs=num_envs,
            seed=seed,
            device=device,
            **self.config
        )

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return 42

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def npc_group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.npc_group_map

    def observation_spec(self, env: EnvBase) -> Composite:
        return Composite(
            shape=(),
            **{k: v for k, v in env.unbatched_observation_spec.items() if k in env.group_map.keys()}
        )

    def action_spec(self, env: EnvBase) -> Composite:
        return Composite(
            shape=(),
            **{k: v for k, v in env.unbatched_action_spec.items() if k in env.group_map.keys()}
        )

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    @staticmethod
    def env_name() -> str:
        return "auction1"

    def log_info(self, batch: TensorDictBase) -> Dict[str, float]:
        return {}
