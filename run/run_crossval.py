import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from experiments.single_step import crossval

def main():
    experiment = crossval.CrossVal('3OEC_current_flow.csv', '5min',
                                   12, 'cuda')
    model = experiment.train_model(100, 'model_1')
    experiment.test_model(model, 'model_1')

if __name__=='__main__':
    main()