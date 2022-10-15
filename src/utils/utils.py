import os
import logging
import dill as pickle


def check_path_exists(path: str = None):
    
    if path is not None and not os.path.exists(path):
        os.makedirs(path)


def check_file_exits(path: str = None):

    if path is not None:
        return os.path.isfile(path)
    return False
    

def save(obj: object, path: str):

    with open(path, 'wb') as fin:
        pickle.dump(obj, fin)
    return path


def load(path: str):

    return pickle.load(open(path, "rb"))
