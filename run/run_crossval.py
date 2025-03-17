import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from experiments.single_step import crossval

def main():
    experiment = crossval.CrossVal('3OEC_current_flow.csv', 50, '5min',
                                   12, 'cuda')
    experiment.run_experiment(2)

if __name__=='__main__':
    main()