#encoding: utf-8

import time
from Client import SocketClient
import json
import random
from distutils.dir_util import copy_tree


# NaiveHunter stratégia implementációja távoli eléréshez.
class MasterClient:

    def __init__(self, name, agents):
        self.name = name
        self.agents = agents
        self.epoch = 1
        self.maps = ["./maps/02_base.txt", "./maps/03_blockade.txt", "./maps/04_mirror.txt"]
        self.map = 0
        self.top_pcs = len(self.agents) * 0.25

    def processObservation(self, fulljson, sendData):
        if fulljson["type"] == "readyToStart":
            print("[" + self.name + "] Starting epoch #" + str(self.epoch))
            sendData(json.dumps({"command": "GameControl", "name": self.name, "payload": {"type": "start", "data": None}}))
            print("[" + self.name + "] Started epoch #" + str(self.epoch))


        if fulljson["type"] == "serverClose":
            for agent in self.agents:
                agent.storeWeights()


        if fulljson["type"] == "leaderBoard":
            print("[" + self.name + "] Finished epoch #" + str(self.epoch))

            time.sleep(0.5)
            print("[" + self.name + "] Saving and calculating datas")

            file = open('data/fitness.txt', 'a')
            file.write("\n" + str(self.epoch))
            for agent in self.agents:
                file.write("," + str(agent.getFitness()))
            file.close()

            agents = self.agents.copy()
            agents.sort(key = lambda x: x.getFitness(), reverse = True)
            
            top_next = 0
            for i in range(len(agents)):
                print(" -> [" + agents[i].name + "] [epoch: #" + str(self.epoch) + "] Fitness score: " + str(agents[i].getFitness()))
                if (i > self.top_pcs):
                    agents[i].setWeights(agents[top_next].getWeights())
                    top_next += 1
                    if top_next > self.top_pcs:
                        top_next = 0
                    agents[i].mutateWeights()
                agents[i].reset()

            self.agents[self.epoch % len(self.agents)].storeWeights()

            # if self.epoch % 200 == 0:
            #     copy_tree("data", "data-" + str(self.epoch))

            if self.epoch % 100 == 0:
                self.map += 1
            if self.map >= len(self.maps):
                self.map = 0

            self.epoch += 1

            sendData(json.dumps({"command": "GameControl", "name": self.name, "payload": {"type": "reset", "data": {"mapPath": self.maps[self.map], "updateMapPath": None}}}))

