import matplotlib.pyplot as plt
import numpy as np
import os

def plot_preds(prediction, truth):
    plt.figure(figsize=(10, 5))
    plt.plot(truth, label='Actual')
    plt.plot(prediction, label='Predicted')
    plt.legend()
    plt.show()

def plot_preds_from_device(prediction, truth):
    plt.figure(figsize=(10, 5))
    plt.plot(truth.cpu().numpy(), label='Actual')
    plt.plot(prediction, label='Predicted')
    plt.legend()
    plt.show()

def plot_preds_from_device(prediction, truth, filename_prefix='plot', top_dir='./out/temp/'):
    if not os.path.exists(top_dir):
        os.makedirs(top_dir)

    save_dir = os.path.join(top_dir, 'plots')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    truth_np = truth.cpu().numpy()
    prediction_np = np.array(prediction)

    plt.figure(figsize=(10, 5))
    plt.plot(truth_np, label='Actual')
    plt.plot(prediction_np, label='Predicted')
    plt.title(filename_prefix)
    plt.legend()

    # Save plot as PNG
    png_filename = os.path.join(save_dir, f'{filename_prefix}.png')
    plt.savefig(png_filename)
    print(f'Plot saved as {png_filename}')

#     # Save plot as LaTeX (PGF backend)
#     plt.savefig(os.path.join(save_dir, f'{filename_prefix}.pgf'))
#     latex_code = f"""\
# \\begin{{figure}}[htbp]
#     \\centering
#     \\input{{{filename_prefix}.pgf}}
#     \\caption{{Actual vs Predicted values}}
#     \\label{{fig:{filename_prefix}}}
# \\end{{figure}}
# """

#     latex_filename = os.path.join(save_dir, f'{filename_prefix}.tex')
#     with open(latex_filename, 'w') as f:
#         f.write(latex_code)

#     print(f'LaTeX code saved as {latex_filename}')

    plt.close()
