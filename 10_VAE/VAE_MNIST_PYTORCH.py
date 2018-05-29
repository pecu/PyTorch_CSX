# Variational Autoencoder for MNIST with PYTORCH

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image

# Custom data read_in
from data_read_in import get_data
import matplotlib.pyplot as plt

class VAE(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(784, 400)
		self.fc21 = nn.Linear(400, 20)
		self.fc22 = nn.Linear(400, 20)
		self.fc3 = nn.Linear(20, 400)
		self.fc4 = nn.Linear(400, 784)

	def encode(self, x):
		h1 = F.relu(self.fc1(x))
		return self.fc21(h1), self.fc22(h1)

	def reparameterize(self, mu, logvar):
		if self.training:
			std = torch.exp(0.5*logvar)
			eps = torch.randn_like(std)
			return eps.mul(std).add_(mu)  # add_ in_place addition
		else:
			return mu

	def decode(self, z):
		h3 = F.relu(self.fc3(z))
		return F.sigmoid(self.fc4(h3))

	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
##
def train_vae(epoch):
	model.train()
	train_loss = 0
	for batch_idx, data in enumerate(train_loader):
		data = data[0]
		optimizer.zero_grad()
		recon_batch, mu, logvar = model(data)
		loss = loss_function(recon_batch, data, mu, logvar)
		loss.backward()
		train_loss += loss.item()
		optimizer.step()
		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader),
				loss.item() / len(data)))

	print('====> Epoch: {} Average loss: {:.4f}'.format(
		  epoch, train_loss / len(train_loader.dataset)))
	return train_loss


def test_vae(epoch):
	model.eval()
	test_loss = 0
	with torch.no_grad():
		for i, data in enumerate(test_loader):
			data = data[0]
			recon_batch, mu, logvar = model(data)
			test_loss += loss_function(recon_batch, data, mu, logvar).item()
			if i == 0:
				n = batch_size
				print(data.size())
				print(recon_batch.size())
				comparison = torch.cat([data.view(batch_size, 1, 28, 28),
									  recon_batch.view(batch_size, 1, 28, 28)])
				save_image(comparison,
						 'results3/reconstruction_' + str(epoch) + '.png', nrow=n)

	test_loss /= len(test_loader.dataset)
	print('====> Test set loss: {:.4f}'.format(test_loss))
	return test_loss




## Load MNIST data
X, ind, Y = get_data()
X_train = torch.from_numpy(X[:30000]).float()
X_test = torch.from_numpy(X[30000:]).float()
batch_size = 5
train = TensorDataset(X_train)
test = TensorDataset(X_test)
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)
##
## Initialize the VAE and optimizer
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_lost_list = []
test_lost_list = []
for epoch in range(1, 11):
	train_loss = train_vae(epoch)
	train_lost_list.append(train_loss)
	test_lost = test_vae(epoch)
	test_lost_list.append(test_lost)
	with torch.no_grad():
		sample = torch.randn(64, 20)
		sample = model.decode(sample)
		save_image(sample.view(64, 1, 28, 28),'results3/sample_' + str(epoch) + '.png')


plt.figure(1)
plt.subplot(211)
plt.plot(train_lost_list)
plt.title('Training lost')
plt.subplot(212)
plt.plot(test_lost_list)
plt.title('Test lost')
plt.show()