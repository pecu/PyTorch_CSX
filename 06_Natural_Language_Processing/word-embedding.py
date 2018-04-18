import torch
from torch import nn
from torch.autograd import Variable

import sys

intValue = int(sys.argv[1])

#using nn.Embedding to get 2*5 vector
embeds = nn.Embedding(2,5)
if intValue == 1:
  print(embeds.weight)

#modify 2*5 vector 
embeds.weight.data = torch.ones(2,5)
if intValue == 2:
  print(embeds.weight)

#get 50th word vector of 100*10 vector
embeds = nn.Embedding(100, 10)
single_word_embed = embeds(Variable(torch.LongTensor([50])))
if intValue == 3:
  print(single_word_embed)
