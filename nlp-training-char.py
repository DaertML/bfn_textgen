import torch
import numpy as np

import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import torch.nn as nn
import torch.nn.functional as F
def round_floats(lst):
  # initialize an empty list to store the rounded integers
  result = []
  # loop through each sublist in the list
  for sublst in lst:
    # initialize an empty list to store the rounded integers for the current sublist
    subresult = []
    # loop through each element in the sublist
    for x in sublst:
      # check if the element is a float
      if isinstance(x, float):
        # round the float to the nearest integer and append it to the subresult list
        subresult.append(round(x))
      else:
        # raise an exception if the element is not a float
        raise TypeError(f"Invalid element: {x}")
    # append the subresult list to the result list
    result.append(subresult)
  # return the result list
  return result

def pad_list_of_lists(list_of_lists, padding_value):
    # Find the maximum length of the sublists
    max_len = max(len(sublist) for sublist in list_of_lists)
    # Create a new list to store the padded sublists
    padded_list_of_lists = []
    # Loop through each sublist
    for sublist in list_of_lists:
        # Copy the sublist as a new list
        padded_sublist = list(sublist)
        # Append the padding value to the end of the sublist until it reaches the maximum length
        while len(padded_sublist) < max_len:
            padded_sublist.append(padding_value)
        # Add the padded sublist to the new list
        padded_list_of_lists.append(padded_sublist)
    # Return the new list
    return padded_list_of_lists

class BayesianFlowNetwork(nn.Module):
    """
    Bayesian Flow Network (BFN) model.
    
    Parameters
    ----------
    D : int, default=2
        Dimensionality of the data.
    K : int, default=2
        Number of classes.
    hidden_dim : int, default=16
        Dimension of the hidden layer.
    """

    def __init__(self, D=2, K=2, hidden_dim=32, beta=3.0):
        super(BayesianFlowNetwork, self).__init__()
        self.beta = beta
        self.D = D
        self.K = K
        # Define the number of output classes based on K
        output_classes = K if K > 2 else 1

        # Define the neural network layers
        self.layer = nn.Sequential(
            nn.Linear(D * K + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, D * output_classes)
        )

    def forward(self, theta, t):
        """
        Forward pass of the Bayesian Flow Network.
        
        Parameters
        ----------
        theta : torch.Tensor
            Tensor of shape (B, D, K).
        t : torch.Tensor
            Tensor of shape (B,).
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, D, K).
        """
        theta = (theta * 2) - 1  # scaled in [-1, 1]
        theta = theta.view(theta.shape[0], -1)  # (B, D * K)
        input_ = torch.cat((theta, t.unsqueeze(-1)), dim=-1)
        output = self.layer(input_)  # (B, D * K)
        output = output.view(output.shape[0], self.D, -1)
        return output

    
    def discrete_output_distribution(self, theta, t):
        """
        Computes the discrete output distribution.

        Algorithm 
        
        Parameters
        ----------
        theta : torch.Tensor
            Input tensor of shape (B, D, K).
        t : torch.Tensor
            Time tensor of shape (B,).
        
        Returns
        -------
        torch.Tensor
            Output probability tensor. For K=2, shape is (B, D, 2). 
            Otherwise, shape is (B, D, K).
        """
        B, D, K = theta.shape
    
        # Get the forward pass output and reshape
        output = self.forward(theta, t)
    
        # Check the number of classes and compute the output probabilities accordingly 
        if K == 2:
            p0_1 = torch.sigmoid(output)  # (B, D, 1)
            p0_2 = 1 - p0_1
            p0 = torch.cat((p0_1, p0_2), dim=-1)  # (B, D, 2)
        else:
            p0 = torch.nn.functional.softmax(output, dim=-1)
        return p0

    def process(self, x):
        B, D = x.shape
        # print(f"B {B}, D {D}")
        
        # Step 1: Sample t from U(0, 1)
        t = torch.rand((x.size(0),), device=x.device, dtype=torch.float32)

        # Step 2: Calculate Beta
        beta = self.beta * (t ** 2)  # (B,)

        # Step 3: Sample y from N(beta * (K * one_hot(X)) 
        one_hot_x = F.one_hot(x, num_classes=self.K).float()  # (B, D, K)
        mean = beta[:, None, None] * (self.K * one_hot_x - 1)
        std = (beta * self.K)[:, None, None].sqrt()
        eps = torch.randn_like(mean)
        y = mean + std * eps

        # Step 4: Compute the Theta
        theta = F.softmax(y, dim=-1)

        # Step 5: Calculate the output distribution
        p_0 = self.discrete_output_distribution(theta, t)  # (B, D, K)

        e_x = one_hot_x
        e_hat = p_0  # (B, D, K)
        L_infinity = self.K * self.beta * t[:, None, None] * ((e_x - e_hat) ** 2)
        return L_infinity.mean()

    @torch.inference_mode()
    def sample(self, batch_size=128, nb_steps=10, device='cpu'):
        self.eval()
        
        # get prior 
        theta = torch.ones((batch_size, self.D, self.K), device=device) / self.K

        for i in range(1, nb_steps+1):
            t = (i-1) / nb_steps
            t = t * torch.ones((theta.shape[0]), device=theta.device, dtype=theta.dtype)
            
            k_probs = self.discrete_output_distribution(theta, t)  # (B, D, K)
            k = torch.distributions.Categorical(probs=k_probs).sample()  # (B, D)
            alpha = self.beta * (2 * i - 1) / (nb_steps ** 2)

            e_k = F.one_hot(k, num_classes=self.K).float()  # (B, D, K)
            mean = alpha * (self.K * e_k - 1)
            var = (alpha * self.K)
            std = torch.full_like(mean, fill_value=var).sqrt()
            eps = torch.randn_like(e_k)
            
            y = mean + std * eps  # (B, D, K)
            
            theta_prime = torch.exp(y) * theta
            theta = theta_prime / theta_prime.sum(-1, keepdim=True)

        k_probs_final = self.discrete_output_distribution(theta, torch.ones_like(t))
        k_final = torch.distributions.Categorical(probs=k_probs_final).sample()

        return k_final
    
def values_to_string(lst):
  # create a dictionary that maps each position in the alphabet to its lowercase letter
  alphabet = {i + 1: chr(ord('a') + i) for i in range(26)}
  # add a special entry for the value 27 with the space character
  alphabet[27] = ' '
  # initialize an empty string to store the result
  result = ''
  # loop through each sublist in the list
  for sublst in lst:
    # loop through each value in the sublist
    for x in sublst:
      # check if the value is an integer between 1 and 27
      if isinstance(x, int) and 1 <= x <= 27:
        # append the corresponding letter to the result string
        result += alphabet[x]
      else:
        # raise an exception if the value is not valid
        continue
        raise ValueError(f"Invalid value: {x}")
    # append a newline character to the result string after each sublist
    result += '\n'
  # return the result string
  return result

def string_to_values(s):
  # create a dictionary that maps each lowercase letter to its position in the alphabet
  alphabet = {chr(ord('a') + i): i + 1 for i in range(26)}
  # add a special entry for the space character with value 27
  alphabet[' '] = 27
  # initialize an empty list to store the values
  values = []
  # loop through each character in the string
  for c in s:
    # convert the character to lowercase
    c = c.lower()
    # check if the character is in the alphabet dictionary
    if c in alphabet:
      # append the corresponding value to the list
      values.append(alphabet[c])
    else:
      # raise an exception if the character is not in the alphabet dictionary
      raise ValueError(f"Invalid character: {c}")
  # return the list of values
  return values

def get_datapoint(batch=1, device='cpu'):
    sentences = [
        "The sky is blue and the sun is shining",
        "She loves to read books and watch movies",
        "He is a good friend and a great teacher",
        "They went to the park and played soccer",
        "I like to eat pizza and ice cream",
        "She has a beautiful voice and a charming smile",
        "He works hard and earns well",
        "They are smart and funny",
        "I enjoy listening to music and playing guitar",
        "She studies hard and gets good grades"
    ]

    data_int = []
    for elem in sentences:
        data_int.append(string_to_values(elem))
    
    data_int = pad_list_of_lists(data_int,0)

    #X = torch.stack([torch.tensor(data_int), torch.tensor(data_int)], dim=0)
    #return X.long().transpose(0, 1)
    return torch.tensor(data_int)#.unsqueeze(0)
    #return X.long().transpose(0, 1)

X = get_datapoint()  # (B, D=2) with K=2 classes 
print(X.shape)
from torch.optim import AdamW
from tqdm.auto import tqdm


# 46 is the length of each vector: 46 chars each
# 28 is the number of possible tokens: 26 letters and 1 space
bfn = BayesianFlowNetwork(D=46, K=28, hidden_dim=100)
optim = AdamW(bfn.parameters(), lr=1e-2)

n = 1000
losses = []
for i in tqdm(range(n)):
    optim.zero_grad()

    X = get_datapoint(device='cpu')
    loss = bfn.process(X)
    loss.backward()

    optim.step()

    losses.append(loss.item())

plt.plot(losses)

x_hat = bfn.sample(device='cpu', nb_steps=100).cpu().numpy()
x_hat = x_hat + (np.random.randn(*x_hat.shape) * 0.1) # add some noise so we can see it

floats = round_floats(x_hat)
for flist in floats:
    print("".join(values_to_string([flist])))
plt.title("Samples")
plt.scatter(x_hat[:, 0], x_hat[:, 1]);
plt.grid()
plt.show()
