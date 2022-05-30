#encoding: utf-8

import time
from Client import SocketClient
import json
from distutils.dir_util import copy_tree


# NaiveHunter stratégia implementációja távoli eléréshez.
class MasterClient:

    def __init__(self, name, agents):
        self.name = name
        self.agents = agents
        self.epoch = 1
        self.maps = ["./maps/02_base.txt", "./maps/03_blockade.txt", "./maps/04_mirror.txt"]
        self.map = 0

    def processObservation(self, fulljson, sendData):
        if fulljson["type"] == "leaderBoard":
            print("[" + self.name + "] Finished epoch #" + str(self.epoch))

            time.sleep(0.5)
            print("[" + self.name + "] Saving and calculating datas")

            file = open('data/fitness.txt', 'a')
            file.write("\n" + str(self.epoch))
            for agent in self.agents:
                file.write("," + str(agent.getFitness()))
            file.close()

            agents = self.agents
            agents.sort(key = lambda x: x.getFitness(), reverse = True)
            
            agents[2].setWeights(agents[0].getWeights())
            agents[2].mutateWeights()
            agents[2].storeWeights()
            agents[3].setWeights(agents[1].getWeights())
            agents[3].mutateWeights()
            agents[3].storeWeights()

            
            if self.epoch % 100 == 0:
                copy_tree("data", "data-" + str(self.epoch))

            if self.epoch % 500 == 0:
                self.map += 1
            if self.map >= len(self.maps):
                self.map = 0

            self.epoch += 1

            sendData(json.dumps({"command": "GameControl", "name": self.name, "payload": {"type": "reset", "data": {"mapPath": self.maps[self.map], "updateMapPath": None}}}))

        if fulljson["type"] == "readyToStart":
            print("[" + self.name + "] Starting epoch #" + str(self.epoch))
            sendData(json.dumps({"command": "GameControl", "name": self.name, "payload": {"type": "start", "data": None}}))
