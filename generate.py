import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from gpt import GPTLanguageModel, decode
from datetime import datetime

model_save_path = "fully_train.pth"

model = GPTLanguageModel()
m = model.to(device)

# Model loading for generation
print("Loading the model for generation...")
checkpoint = torch.load(model_save_path, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set the model to evaluation mode

# Generate from the loaded model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print(generated_text)

# Save the generated text to a file if needed
output_file = 'generated_text.txt'
with open(output_file, 'a', encoding='utf-8') as f:  # 'a' mode for appending
    f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    f.write('\n') 
    f.write(generated_text)
    f.write('\n')  # Add a newline for separation if needed
print(f"Generated text appended to {output_file}")
