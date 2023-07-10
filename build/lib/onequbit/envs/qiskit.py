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
import numpy as np

class QiskitEnv(gym.Env):
    def __init__(self, size=1):
        self.size = size  # The size of the circuit
        self.circuit = QuantumCircuit(self.size)
        self.number_qubits = 0

        # We have 4 different gates
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: XGate(),
            1: YGate(),
            2: ZGate(),
            3: HGate(),
        }
        
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=0, high=1, shape=(1,2), dtype=int),
                "target": spaces.Box(low=0, high=1, shape=(1,2), dtype=int),
            }
        )

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {
            "qubits": self.number_qubits,
            }


    def _get_values(self):
        sim = Aer.get_backend('statevector_simulator')
        result = sim.run(self.circuit, shots=1).result().get_statevector()
        return Statevector(result).probabilities()
    
    def reset(self):
        self.circuit = QuantumCircuit(self.size)
        self.number_qubits = 0
        self._agent_location = np.array([1,0])
        self._target_location = np.array([0,1])

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        self.circuit.append(self._action_to_direction[action], [0])
        self._agent_location = np.array(self._get_values())
        self.number_qubits += 1

        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = self._agent_location[1]
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
    
    def render(self):
        return self.circuit.draw(output="mpl")
