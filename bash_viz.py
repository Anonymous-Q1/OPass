import sys
from Autotuning.util import viz2file

def main():
    filePath = sys.argv[1]
    viz2file(filePath)

if __name__ == '__main__':
    main()