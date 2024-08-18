import torch
import numpy as np
import torch.utils

size = 5

x_mat = torch.randn(size , size)

base_matrix = torch.tril(torch.ones(size , size))
temp = torch.zeros((size , size))
masked_mat = temp.masked_fill(base_matrix == 0 , float('-inf'))

masked_mat = torch.nn.functional.softmax(masked_mat , dim=-1)

aggregated = x_mat @ masked_mat

print(aggregated)