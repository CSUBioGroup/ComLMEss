import torch
import torch.nn as nn
from modules import ComOutput, ComIntermediate, MLP_output, ConvFeatureExtractor5


class ComLMEss(nn.Module):
    def __init__(self, conv_input=1024, nhead=2, dropout=0.3, activation="relu", kernel_size=4, stride=4, seq_len=400,
                 d_att=256, kernel_size_onto=5, kernel_size_prot=5, kernel_size_esm=5, num_filters=64):
        super(ComLMEss, self).__init__()

        self.onto_conv = ConvFeatureExtractor5(input_dim=conv_input, kernel_size=kernel_size_onto, num_filters=num_filters)
        self.prot_conv = ConvFeatureExtractor5(input_dim=conv_input, kernel_size=kernel_size_prot, num_filters=num_filters)
        self.esm_conv = ConvFeatureExtractor5(input_dim=conv_input, kernel_size=kernel_size_esm, num_filters=num_filters)

        seq_len /= 4

        self.onto_attention = nn.MultiheadAttention(d_att, nhead, dropout=dropout)
        self.prot_attention = nn.MultiheadAttention(d_att, nhead, dropout=dropout)
        self.esm_attention = nn.MultiheadAttention(d_att, nhead, dropout=dropout)

        self.onto_selfoutput1 = ComOutput(d_model=d_att, dropout=dropout)
        self.prot_selfoutput1 = ComOutput(d_model=d_att, dropout=dropout)
        self.esm_selfoutput1 = ComOutput(d_model=d_att, dropout=dropout)

        self.onto_intermediate = ComIntermediate(d_model=d_att, activation=activation)
        self.prot_intermediate = ComIntermediate(d_model=d_att, activation=activation)
        self.esm_intermediate = ComIntermediate(d_model=d_att, activation=activation)

        self.onto_selfoutput2 = ComOutput(d_model=d_att, dropout=dropout)
        self.prot_selfoutput2 = ComOutput(d_model=d_att, dropout=dropout)
        self.esm_selfoutput2 = ComOutput(d_model=d_att, dropout=dropout)

        self.onto_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)
        self.prot_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)
        self.esm_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)

        pooled_length = (seq_len - kernel_size) // stride + 1
        input_dim = d_att * 3 * pooled_length
        self.mlp = MLP_output(input_dim, output_dim=1, dropout=dropout)

    def forward(self, onto, prot, esm):
        onto = self.onto_conv(onto)
        prot = self.prot_conv(prot)
        esm = self.esm_conv(esm)

        onto = onto.permute(1, 0, 2)
        prot = prot.permute(1, 0, 2)
        esm = esm.permute(1, 0, 2)

        onto_attention_score, onto_attention_weights = self.onto_attention(onto, onto, onto)
        onto_attention_score = self.onto_selfoutput1(onto_attention_score, onto)
        onto_attention_score1 = self.onto_intermediate(onto_attention_score)
        onto_attention_score = self.onto_selfoutput2(onto_attention_score1, onto_attention_score)

        prot_attention_score, prot_attention_weights = self.prot_attention(prot, prot, prot)
        prot_attention_score = self.prot_selfoutput1(prot_attention_score, prot)
        prot_attention_score1 = self.prot_intermediate(prot_attention_score)
        prot_attention_score = self.prot_selfoutput2(prot_attention_score1, prot_attention_score)

        esm_attention_score, esm_attention_weights = self.esm_attention(esm, esm, esm)
        esm_attention_score = self.esm_selfoutput1(esm_attention_score, esm)
        esm_attention_score1 = self.esm_intermediate(esm_attention_score)
        esm_attention_score = self.esm_selfoutput2(esm_attention_score1, esm_attention_score)

        onto_before_pool = onto_attention_score.permute(1, 2, 0)
        prot_before_pool = prot_attention_score.permute(1, 2, 0)
        esm_before_pool = esm_attention_score.permute(1, 2, 0)

        onto_pool = self.onto_pool(onto_before_pool)
        prot_pool = self.prot_pool(prot_before_pool)
        esm_pool = self.esm_pool(esm_before_pool)

        onto_pool.permute(0, 2, 1)
        prot_pool.permute(0, 2, 1)
        esm_pool.permute(0, 2, 1)

        all_seq = torch.cat((prot_pool, esm_pool, onto_pool), dim=-1)
        all_flattened_seq = all_seq.view(all_seq.size(0), -1)

        output = self.mlp(all_flattened_seq)
        return output

