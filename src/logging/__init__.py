import torch, os, json
def log_memory(log_folder:str, filename:str):
  if not os.path.exists(log_folder):
    os.makedirs(log_folder)
    
  with open(f"{log_folder}/{filename}", mode='w') as f:
    json.dump(torch.cuda.memory_stats(), f, indent=2)