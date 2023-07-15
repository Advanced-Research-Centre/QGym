#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Kajetan Knopp <kajetan@knopp.com.pl>
#
# Distributed under terms of the MIT license.

"""
Init file for the package
"""
from gymnasium.envs.registration import register

register(
    id='QGym/Qiskit-v0',
    entry_point='QGym.envs:QiskitEnv'
)
