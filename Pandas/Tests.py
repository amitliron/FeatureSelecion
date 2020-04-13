
from Configuration.ConfigurationManager import ConfigurationManager


def main():
    mgr = ConfigurationManager()
    mgr = ConfigurationManager()
    config = mgr.get_configuration()

    if config['Dataset']['iris'] == "True":
        print("True")
    else:
        print("False")

    None
    mgr = ConfigurationManager()


if __name__ == "__main__":
    main()