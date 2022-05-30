import time
import json
from Client import SocketClient
from Config import *
from Keras_Client_Main import RemoteNaiveHunterStrategy
from Master_Client_Main import MasterClient

if __name__ == "__main__":
    n = NUM_PLAYERS

    agents = []
    clients = []
    for i in range(n):
        agents.append(RemoteNaiveHunterStrategy("agent" + str(i), "data"))
        clients.append(SocketClient("localhost", 42069, agents[i].processObservation))


    master = MasterClient("master", agents)
    master_client = SocketClient("localhost", 42069, master.processObservation)


    for client in clients:
        client.start()

    master_client.start()


    # Kis szünet, hogy a kapcsolat felépülhessen, a start nem blockol, a kliens külső szálon fut
    time.sleep(0.1)
    # Regisztráció a megfelelő névvel
    for i in range(n):
        clients[i].sendData(json.dumps({"command": "SetName", "name": agents[i].name, "payload": None}))

    master_client.sendData(json.dumps({"command": "SetName", "name": "master", "payload": None}))

