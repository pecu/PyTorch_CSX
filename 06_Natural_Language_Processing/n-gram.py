import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys

intValue = int(sys.argv[1])

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

#show test_sentence
if intValue == 1:
  print(test_sentence)
  print(len(test_sentence))
  sys.exit(0)

trigram = [((test_sentence[i], test_sentence[i+1]), test_sentence[i+2])
           for i in range(len(test_sentence)-2)]

#show ((word1, word2), word3)
if intValue == 2:
  print(len(trigram))
  print(trigram[0])
  sys.exit(0)

vocb = set(test_sentence)
word_to_idx = {word: i for i, word in enumerate(vocb)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}

#show encoded dictory
if intValue == 3:
  #print(word_to_idx)
  print(idx_to_word)
  sys.exit(0)

class NgramModel(nn.Module):
  def __init__(self, vocb_size, context_size=CONTEXT_SIZE, n_dim=EMBEDDING_DIM):
    super(NgramModel, self).__init__()
    self.n_word = vocb_size
    self.embedding = nn.Embedding(self.n_word, n_dim)
    self.linear1 = nn.Linear(context_size*n_dim, 128)
    self.linear2 = nn.Linear(128, self.n_word)

  def forward(self, x):
    emb = self.embedding(x)
    emb = emb.view(1, -1)
    out = self.linear1(emb)
    out = F.relu(out)
    out = self.linear2(out)
    log_prob = F.log_softmax(out)
    return log_prob

ngrammodel = NgramModel(len(word_to_idx))
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(ngrammodel.parameters(), lr=1e-3)

for epoch in range(250):
  print('epoch: {}'.format(epoch+1))
  print('*'*10)
  running_loss = 0

  for data in trigram:
    word, label = data
    word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
    label = Variable(torch.LongTensor([word_to_idx[label]]))
    #forward
    out = ngrammodel(word)
    loss = criterion(out, label)
    running_loss += loss.data[0]
    #backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print('Loss: {:.6f}'.format(running_loss / len(word_to_idx)))

word, label = trigram[3]
word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
out = ngrammodel(word)
_, predict_label = torch.max(out, 1)
predict_word = idx_to_word[predict_label.data[0]]
print(trigram[3])
print('real word is {}, predict word is {}'.format(label, predict_word))
