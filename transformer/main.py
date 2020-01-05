from config import get_args
from Dataset import Dataset



def main():
    config = get_args()
    dataset = Dataset(config)


if __name__ == '__main__' :
    main()