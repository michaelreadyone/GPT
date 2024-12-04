import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from gpt import GPTLanguageModel, decode
from datetime import datetime
import time
import matplotlib.pyplot as plt
from icecream import ic

model_save_path = "fully_train.pth"

model = GPTLanguageModel()
m = model.to(device)

# Model loading for generation
print("Loading the model for generation...")
checkpoint = torch.load(model_save_path, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set the model to evaluation mode

if __name__ == '__main__':
    
    max_new_tokens = 500
    durations = []
    durations_diff = []
    # context = torch.zeros((1, 1), dtype=torch.long, device=device)
    context = torch.tensor([[ 0,  0, 15, 27, 25, 21, 26, 21, 33, 31, 10,  0, 27,  6,  1, 61, 43, 50,
                  50,  6,  1, 61, 43, 50, 50,  6,  1, 40, 43, 50, 47, 43, 60, 43,  1, 58,
                  46, 43,  1, 53, 40, 43, 63,  1, 61, 43, 56, 43,  1, 58, 46, 56, 59, 57,
                  58, 43, 58, 46,  1, 39, 50, 50,  8,  0,  0, 25, 17, 26, 17, 26, 21, 33,
                  31, 10,  0, 26, 53,  1, 51, 53, 56, 43,  6,  1, 51, 63,  1, 50, 53, 56,
                  42, 10,  1, 50, 43, 58,  5, 57,  1, 43, 52, 58, 56, 43, 39, 58,  1, 63,
                  53, 59,  1, 46, 39, 60, 43,  0, 39, 41, 41, 59, 57, 39, 58, 47, 53, 52,
                   0, 37, 53, 59, 56, 57, 43, 50, 44,  1, 44, 53, 56,  1, 63, 53, 59, 56,
                   1, 45, 53, 53, 42,  1, 61, 53, 56, 42, 57,  8,  0,  0, 15, 27, 30, 21,
                  27, 24, 13, 26, 33, 31, 10,  0, 21, 58,  1, 47, 57,  1, 39,  1, 60, 43,
                  56, 63,  1, 42, 56, 43, 39, 51,  1, 58, 53,  1, 42, 53,  1, 63, 53, 59,
                   1, 54, 56, 53, 51, 47, 57, 43, 42, 11,  0, 26, 53, 58,  1, 57, 43, 50,
                  44, 10,  1, 52, 53, 56,  6,  1, 39, 57,  1, 58, 46, 43,  1, 41, 56, 59,
                  43, 50,  1, 47, 57,  1, 63, 53, 59, 56, 57,  6,  0, 32, 53,  1, 57, 39,
                  63,  1]], device=device)
    ic(context.shape)
    ic(f'prefix: {decode(context[-1].tolist())}')
    res, times_per_token = model.generate(context, max_new_tokens)
    # ic(res)
    ic(decode(res[-1].tolist()))
    # ic(tokenizer.decode(res[-1]))
    # Plot the list
    # plt.plot(times_per_token, marker='o', linestyle='-', color='b', label='Data Points')

    # # Adding labels and title
    # plt.title('Plot of the Python List')
    # plt.xlabel('Index')
    # plt.ylabel('Value')

    # # Show the legend
    # plt.legend()

    # # Display the plot
    # plt.show()
