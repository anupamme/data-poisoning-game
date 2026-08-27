import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Dict

from .attack_strategies import AttackStrategy, PoisonedDataset, backdoor_pixel_fn


class ConsensusShiftAttack(AttackStrategy):
    """Two-phase attack that gradually shifts the trimmed-mean consensus before injecting a backdoor.

    Phase 1: Submit updates biased toward the target direction at a small fraction
    of the honest update magnitude, staying within the trimmed-mean acceptance range.

    Phase 2: After the consensus has shifted sufficiently, submit full backdoor updates
    that now appear normal relative to the shifted reference.
    """

    def __init__(self, shift_rate: float = 0.1, shift_rounds: int = 25,
                 trigger_size: int = 4, target_class: int = 0,
                 poison_fraction: float = 0.5, scale_factor: float = 1.0):
        super().__init__("consensus_shift")
        self.shift_rate = shift_rate
        self.shift_rounds = shift_rounds
        self.trigger_size = trigger_size
        self.target_class = target_class
        self.poison_fraction = poison_fraction
        self.scale_factor = scale_factor
        self._round = 0
        self._target_direction = None

    def poison_dataset(self, dataset) -> Dataset:
        return PoisonedDataset(
            dataset,
            backdoor_pixel_fn(self.trigger_size, self.target_class),
            self.poison_fraction,
        )

    def manipulate_update(self, update: Dict[str, torch.Tensor],
                          global_model: nn.Module) -> Dict[str, torch.Tensor]:
        self._round += 1
        keys = list(update.keys())

        if self._round <= self.shift_rounds:
            return self._phase1_shift(update, keys)
        else:
            return self._phase2_inject(update, keys)

    def _phase1_shift(self, update: Dict[str, torch.Tensor],
                      keys: list) -> Dict[str, torch.Tensor]:
        if self._target_direction is None:
            self._target_direction = {k: v.clone() for k, v in update.items()}

        direction_flat = torch.cat([self._target_direction[k].flatten() for k in keys])
        update_flat = torch.cat([update[k].flatten() for k in keys])

        update_norm = update_flat.norm().item()
        dir_norm = direction_flat.norm().item()
        if dir_norm < 1e-8:
            return update

        scaled_direction = direction_flat * (self.shift_rate * update_norm / dir_norm)
        shifted_flat = update_flat + scaled_direction

        result = {}
        offset = 0
        for k in keys:
            numel = update[k].numel()
            result[k] = shifted_flat[offset:offset + numel].reshape(update[k].shape)
            offset += numel
        return result

    def _phase2_inject(self, update: Dict[str, torch.Tensor],
                       keys: list) -> Dict[str, torch.Tensor]:
        if self.scale_factor == 1.0:
            return update
        return {k: v * self.scale_factor for k, v in update.items()}

    @property
    def cost(self) -> float:
        return 0.35


# To register this attack, add to ATTACK_REGISTRY in attack_strategies.py:
#   "consensus_shift": ConsensusShiftAttack,
