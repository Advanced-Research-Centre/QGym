#/usr/bin/env python3
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
        self.size = size
        self.circuit = QuantumCircuit(self.size)
        self.number_gates = 0

        # We have 4 different gates

        
        self._available_gates = [XGate, YGate, ZGate, HGate, SGate, TGate, IGate]
        
        self.action_space = spaces.Discrete(len(self._available_gates)*self.size)

        self._gate_definition = {
            j*self.size+i: gate for i in range(self.size) for j, gate in enumerate(self._available_gates)
        }
        
        self.observation_space = spaces.Box(0.0, 1.0, (1, 2**self.size), dtype=np.float64)
    
    def _get_obs(self):
        return self._agent_location
    
    def _get_info(self):
        return {"gates": self.number_gates}


    def _get_values(self):
        sim = Aer.get_backend('statevector_simulator')
        result = sim.run(self.circuit, shots=1).result().get_statevector()
        return Statevector(result).probabilities()
    
    def reset(self):
        self.circuit = QuantumCircuit(self.size)
        self.number_qubits = 0
        self._agent_location = [1]+(2**self.size-1)*[0]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        self.circuit.append(self._gate_definition[action](), [action%self.size])
        self._agent_location = self._get_values()
        self.number_gates += 1

        terminated = (self._agent_location[-1] > 0.95)
        reward = self._agent_location[-1]
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
