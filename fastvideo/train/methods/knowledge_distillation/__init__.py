# SPDX-License-Identifier: Apache-2.0

from fastvideo.train.methods.knowledge_distillation.kd import (
    KDCausalMethod,
    KDMethod,
)
from fastvideo.train.methods.knowledge_distillation.minimax_h3_velocity import (
    MiniMaxH3VelocityKDMethod, )

__all__ = ["KDCausalMethod", "KDMethod", "MiniMaxH3VelocityKDMethod"]
