import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from experiments.single_step import transfer_learning

def main():
    experiment = transfer_learning.TransferLearning('3OEC_current_flow.csv', 100, '5min',
                                   12, 'mps')
    save_dir = f"./out/tl_100epochs_10runs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    experiment.run_experiment(10, save_dir)

if __name__=='__main__':
    main()