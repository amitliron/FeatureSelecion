#from Assets import Singleton
from Assets.Singleton import Singleton


class ConfigurationManager(metaclass=Singleton):

    def __init__(self):
        self.__read_configuration()

    def __read_configuration(self):
        print("read...")
        import configparser
        self.__config = configparser.ConfigParser()
        self.__config.read('../Configuration/Configuration.ini')

    def get_configuration(self):
        return self.__config