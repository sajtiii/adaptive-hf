#encoding: utf-8

import time
from unittest import findTestCases
from Client import SocketClient
import json
import numpy as np
from os import path

# General Dense Layer template for hidden layers
class DenseLayer: 
    def __init__(self, inputs, neurons):
        self.inputs = inputs
        self.neurons = neurons
        self.weights = 0.1 * np.random.randn(self.inputs, self.neurons)
        self.biases = np.zeros((1, self.neurons))

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

    def mutate(self):
        self.weights += np.random.randn(self.inputs, self.neurons) * np.random.randn(self.inputs, self.neurons)
        self.biases += 0.01 * np.random.randn(1, self.neurons)


class ReLU:
    def forward(self, inputs):
        return np.maximum(0, inputs)

class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        return exp_values / np.sum(exp_values, axis = 1, keepdims = True)


# NaiveHunter stratégia implementációja távoli eléréshez.
class RemoteNaiveHunterStrategy:
    def __init__(self, name, data_folder):
        self.name = name
        self.data_folder = data_folder

        self.size = 0
        self.lastpos = None
        self.stays = 0

        self.layers = [ DenseLayer(400, 200), DenseLayer(200, 54), DenseLayer(54, 9) ]
        self.activations = [ ReLU(), ReLU(), Softmax() ]

    def forward(self, inputs):
        output = inputs
        for i in range(len(self.layers)):
            output = self.layers[i].forward(output)
            output = self.activations[i].forward(output)

        return output[0]

    def getWeights(self):
        weights = {"weights": [], "biases": []}
        for i in range(len(self.layers)):
            weights["weights"].append(self.layers[i].weights)
            weights["biases"].append(self.layers[i].biases)
        return weights

    def setWeights(self, weights):
        print(" -> [" + self.name + "] Setting weights")
        for i in range(len(self.layers)):
            self.layers[i].weights = weights["weights"][i]
            self.layers[i].biases = weights["biases"][i]
        print(" -> [" + self.name + "] Setted weights")

    def loadWeights(self):
        for i in range(len(self.layers)):
            if path.exists(self.data_folder + "/" + self.name + "_w" + str(i) + ".txt"):
                print(" -> [" + self.name + "] Loading weights for w" + str(i))
                self.layers[i].weights = np.loadtxt(self.data_folder + "/" + self.name + "_w" + str(i) + ".txt")
                print(" -> [" + self.name + "] Loaded weights for w" + str(i))

    def storeWeights(self):
        print(" -> [" + self.name + "] Storing weights")
        for i in range(len(self.layers)):
            np.savetxt(self.data_folder + "/" + self.name + "_w" + str(i) + ".txt", self.layers[i].weights)
        print(" -> [" + self.name + "] Stored weights")

    def mutateWeights(self):
        print(" -> [" + self.name + "] Mutating weights")
        for layer in self.layers:
            layer.mutate()
        print(" -> [" + self.name + "] Mutated weights")

    def getFitness(self):
        return self.size - 5 - 0.1 * self.stays

    def getInputs(self, jsonData):
        array_locations =   [
                                [-5, 0],   [-3, 4],   [-2, 4],   [-1, 4],   [0, 4],   [1, 4],   [2, 4],   [3, 4],   [0, 5],
                                [-4, 3],   [-3, 3],   [-2, 3],   [-1, 3],   [0, 3],   [1, 3],   [2, 3],   [3, 3],   [4, 3],
                                [-4, 2],   [-3, 2],   [-2, 2],   [-1, 2],   [0, 2],   [1, 2],   [2, 2],   [3, 2],   [4, 2],
                                [-4, 1],   [-3, 1],   [-2, 1],   [-1, 1],   [0, 1],   [1, 1],   [2, 1],   [3, 1],   [4, 1],
                                [-4, 0],   [-3, 0],   [-2, 0],   [-1, 0],             [1, 0],   [2, 0],   [3, 0],   [4, 0],
                                [-4, -1],  [-3, -1],  [-2, -1],  [-1, -1],  [0, -1],  [1, -1],  [2, -1],  [3, -1],  [4, -1],
                                [-4, -2],  [-3, -2],  [-2, -2],  [-1, -2],  [0, -2],  [1, -2],  [2, -2],  [3, -2],  [4, -2],
                                [-4, -3],  [-3, -3],  [-2, -3],  [-1, -3],  [0, -3],  [1, -3],  [2, -3],  [3, -3],  [4, -3],
                                [0, -5],   [-3, -4],  [-2, -4],  [-1, -4],  [0, -4],  [1, -4],  [2, -4],  [3, -4],  [5, 0],
                            ]
        inputs = []
        for index in range(80):
            for vision in jsonData["vision"]:
                if vision["relative_coord"][0] == array_locations[index][0] and vision["relative_coord"][1] == array_locations[index][1]:
                    inputs.append( int(vision["value"] == 9) )
                    inputs.append( int(vision["value"] == 1) )
                    inputs.append( int(vision["value"] == 2) )
                    inputs.append( int(vision["value"] == 3 or (vision["player"] != None and vision["player"]["size"] < jsonData["size"])) )
                    inputs.append( int(vision["value"] == 0 and vision["player"] != None and vision["player"]["size"] >= jsonData["size"]) )
                    continue
        
        return inputs


    # Az egyetlen kötelező elem: A játékmestertől jövő információt feldolgozó és választ elküldő függvény
    def processObservation(self, fulljson, sendData):
        # Játék rendezéssel kapcsolatos üzenetek lekezelése
        if fulljson["type"] == "leaderBoard":
            self.storeWeights()

            for score in fulljson["payload"]["players"]:
                if (score["name"] == self.name):
                    self.size = score["maxSize"]

        if fulljson["type"] == "started":
            self.loadWeights()
            print(" -> [" + self.name + "] Startup message from server")
            

        # Akció előállítása bemenetek alapján (egyezik a NaiveHunterBot-okéval)
        elif fulljson["type"] == "gameData":
            jsonData = fulljson["payload"]
            if "pos" in jsonData.keys() and "tick" in jsonData.keys() and "active" in jsonData.keys() and "size" in jsonData.keys() and "vision" in jsonData.keys():
                
                if self.lastpos is not None:
                    if tuple(self.lastpos) == tuple(jsonData["pos"]):
                        self.stays += 1
                if jsonData["active"]:
                    self.lastpos = jsonData["pos"].copy()


                out = self.forward(self.getInputs(jsonData))
                out_max = max(out)

                actx = "0"
                acty = "0"
                for i in range(3):
                    for j in range(3):
                        if (out[j * 3 + i] == out_max):
                            if (i == 0):
                                actx = "-"
                            if (i == 2):
                                actx = "+"
                            if (j == 0):
                                acty = "-"
                            if (j == 2):
                                acty = "+"
                            break

                # Akció JSON előállítása és elküldése
                action = json.dumps({"command": "SetAction", "name": self.name, "payload": (actx + acty)})
                sendData(action)

