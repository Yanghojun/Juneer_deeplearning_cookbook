import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.process_argument import get_args
from utils.process_configuration import ConfigurationParameters
from data_loader.las_data_loader import LasDataLoader
from models.autoencoder import AutoEncoder

def main():
    # try:
    # capture the command line arguments from the interface script
    args = get_args()

    # pares the configuration parameters for the autoencoder model
    config = ConfigurationParameters(args)

    # except:
    #     print('Missing or Invalid arguments! ')
    #     exit(0)

    dataset = LasDataLoader(config)
    
    model = AutoEncoder(config, dataset)

if __name__ == '__main__':
    main()