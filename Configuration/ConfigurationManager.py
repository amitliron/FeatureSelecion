from Assets import Singleton


class ConfigurationManager(metaclass=Singleton):

    def __init__(self):
        self.read_configuration()

    def __read_configuration(self):
        import configparser
        self.__config = configparser.ConfigParser()
        self.__config.read('../Configuration/Configuration.ini')

    def get_configuration(self):
        return self.__config