import argparse

from utils.config_args import parse_args

"""
Main Framework
"""

def run():
    parser = argparse.ArgumentParser(description="Run Tranformer models.")
    args = parse_args(parser)

    print(args)

if __name__ == "__main__":
    run()