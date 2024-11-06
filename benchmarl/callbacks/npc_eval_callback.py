from typing import List

from benchmarl.experiment.callback import Callback
from tensordict import TensorDictBase
import torch


# Based on Logger.log_evaluation


def get_global_done(td: TensorDictBase):
    return td.get(("next", "done"))


def get_reward(group: str, td: TensorDictBase) -> torch.Tensor:
    return td.get(("next", group, "reward"), None)


class NpcEvalCallback(Callback):
    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        if not len(rollouts):
            return

        max_length_rollout_0 = 0
        for i in range(len(rollouts)):
            r = rollouts[i]
            next_done = get_global_done(r).squeeze(-1)
            done_index = next_done.nonzero(as_tuple=True)[0]
            if done_index.numel() > 0:
                done_index = done_index[0]
                r = r[: done_index + 1]
            if i == 0:
                max_length_rollout_0 = max(r.batch_size[0], max_length_rollout_0)
            rollouts[i] = r

        to_log = {}
        for npc_group in self.experiment.task.npc_group_map(self.experiment.test_env).keys():
            returns = torch.stack(
                [get_reward(npc_group, td).sum(0).mean() for td in rollouts],
                dim=0,
            )

            to_log.update({
                f"eval/{npc_group}/reward/episode_reward_min": returns.min().item(),
                f"eval/{npc_group}/reward/episode_reward_mean": returns.mean().item(),
                f"eval/{npc_group}/reward/episode_reward_max": returns.max().item(),
            })

        self.experiment.logger.log(to_log, step=self.experiment.n_iters_performed)
