#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Kajetan Knopp <kajetan@knopp.com.pl>
#
# Distributed under terms of the MIT license.

"""
Environment for Qiskit
"""
from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import XGate, YGate, ZGate, HGate, SGate, TGate, IGate
import gymnasium as gym
from gymnasium import spaces

class QiskitEnv(gym.Env):
    def __init__(self, size=1):
        self.size = size  # The size of the circuit
        self.circuit = QuantumCircuit(self.size)

        # We have 4 different gates
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: XGate(),
            1: YGate(),
            2: ZGate(),
            3: HGate(),
        }
    
    def get_score(self):
        sim = Aer.get_backend('statevector_simulator')
        result = sim.run(self.circuit, shots=1).result().get_statevector()
        return Statevector(result).probabilities()[1]
    
    def reset(self):
        self.circuit = QuantumCircuit(self.size)
        
        return self.get_score()
    
    def step(self, action):
        self.circuit.append(self._action_to_direction[action], [0])
        return self.get_score(), self.get_score()
    
    def render(self):
        return self.circuit.draw(output="mpl")
