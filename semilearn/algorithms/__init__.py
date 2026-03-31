# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.fixmatch_Shape_Consistency_Loss import FixMatch_Shape_Consistency_Loss
from semilearn.algorithms.fixmatch_Shape_Consistency_Loss_Overlay2Img import Fixmatch_Shape_Consistency_Loss_Overlay2Img
from semilearn.algorithms.fixmatch_Overlay2Img import Fixmatch_Overlay2Img

name2alg = ALGORITHMS

def get_algorithm(args, net_builder, tb_log, logger):
    if args.algorithm in ALGORITHMS:
        alg = ALGORITHMS[args.algorithm]( # name2alg[args.algorithm](
            args=args,
            net_builder=net_builder,
            tb_log=tb_log,
            logger=logger
        )
        return alg
    else:
        raise KeyError(f'Unknown algorithm: {str(args.algorithm)}')



