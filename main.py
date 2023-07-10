#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Kajetan Knopp <kajetan@knopp.com.pl>
#
# Distributed under terms of the MIT license.

"""
Main file for Gym implementation of machine learning for Qiskit quantum 
circuits enironnment.
"""

import gym
import qiskit

circuit = qiskit.QuantumCircuit(1)

circuit.draw(output='mpl')
