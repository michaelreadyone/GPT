import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from datetime import datetime
import time
import matplotlib.pyplot as plt
from icecream import ic

# from gpt_mini import GPTLanguageModel, decode
# model_save_path = "gpt_mini.pth"

from gpt_mini_kv_cache import GPTLanguageModel, decode
model_save_path = "gpt_mini_cache.pth"


model = GPTLanguageModel()
m = model.to(device)

# Model loading for generation
print("Loading the model for generation...")
checkpoint = torch.load(model_save_path, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set the model to evaluation mode

if __name__ == '__main__':
    
    max_new_tokens = 2000
    durations = []
    durations_diff = []
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    start_time = time.time()
    res, times_per_token = model.generate(context, max_new_tokens)
    ic(f'generation takes {time.time() - start_time} seconds')
    ic(res)
    # ic(decode(res[-1].tolist()))

    # # Plot the list
    # plt.plot(times_per_token, marker='o', linestyle='-', color='b', label='Data Points')

    # # Adding labels and title
    # plt.title('Plot of the Python List')
    # plt.xlabel('Index')
    # plt.ylabel('Value')

    # # Show the legend
    # plt.legend()

    # # Display the plot
    # plt.show()
