import torch
import torch.nn.functional as F

# Token values and keys
values = torch.tensor([[1.,0.], [0.,1.], [1.,1.]])
keys   = torch.tensor([[1.,0.], [0.,1.], [1.,1.]])
query  = torch.tensor([1.,0.])  # which token we are focusing on

# Attention score = query · key (dot product)
scores = torch.matmul(keys, query)  
weights = F.softmax(scores, dim=0)  # normalize to probabilities

# Weighted sum of values
attention_output = torch.sum(weights.unsqueeze(1) * values, dim=0)

print("Attention weights:", weights)
print("Attention output:", attention_output)
