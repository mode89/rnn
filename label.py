import argparse

def main():
    args = parse_arguments()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    return parser.parse_args()

if __name__ == "__main__":
    main()
