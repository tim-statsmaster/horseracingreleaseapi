from pymongo import MongoClient

mongo_connection_params = {
    'username': 'root',
    'password': 'YPXvz6vrsKQ8TNNa',
    'host': '3.114.25.239',
    'port': 27017
}

class HorseRacingDB:
    def __init__(self, connection_params=mongo_connection_params):
        username = mongo_connection_params['username']
        password = mongo_connection_params['password']
        host = mongo_connection_params['host']
        port = mongo_connection_params['port']
        connection_string = f"mongodb://{username}:{password}@{host}:{port}"
        self.conn = MongoClient(connection_string)
