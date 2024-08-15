import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}




class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias



class ComOutput(nn.Module):
    def __init__(self, d_model, dropout=0.35):
        super(ComOutput, self).__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.LayerNorm = BertLayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ComIntermediate(nn.Module):
    def __init__(self, d_model, activation="relu"):
        super(ComIntermediate, self).__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.intermediate_act_fn = ACT2FN[activation]

    def forward(self, x):
        x = self.dense(x)
        x = self.intermediate_act_fn(x)
        return x


class ConvFeatureExtractor5(nn.Module):
    def __init__(self, input_dim, num_filters=64, kernel_size=5, stride=2, padding=2):
        super(ConvFeatureExtractor5, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, num_filters*4, kernel_size=kernel_size, stride=stride, padding=padding)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        return x



class MLP_output(nn.Module):
    def __init__(self, input_dim, output_dim=1, dropout=0.1):
        super(MLP_output, self).__init__()
        self.fc = nn .Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x