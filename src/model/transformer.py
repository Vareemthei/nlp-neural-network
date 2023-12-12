"""
This file contains the Transformer class, which is responsible for implementing
the transformer model.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(
            0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]  # (seq_len, d_model)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # model parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout

        # linear layers
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, input, batch_size) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, d_k)
        """
        return input.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def _scaled_dot_product_attention(self, Q, K, V, mask=None) -> torch.Tensor:
        """
        Compute the scaled dot product attention.
        """
        attention = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # mask attention
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -float('inf'))

        weights = F.softmax(attention.float(), dim=-1).type_as(attention)
        weights = F.dropout(weights, p=self.dropout, training=self.training)

        return torch.matmul(weights, V)

    def _combine_heads(self, input, batch_size) -> torch.Tensor:
        """
        Transpose and then concatenate the last two dimensions.
        """
        return input.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

    def forward(self, input, mask=None) -> torch.Tensor:
        """
        Compute the multi-head attention.
        """
        batch_size, seq_len, d_model = input.size()

        assert d_model == self.d_model, "input tensor must have the same dim as d_model"

        Q = self._split_heads(self.W_q(input), batch_size)
        K = self._split_heads(self.W_k(input), batch_size)
        V = self._split_heads(self.W_v(input), batch_size)

        attn_output = self._scaled_dot_product_attention(Q, K, V, mask)
        attn_output = self._combine_heads(attn_output, batch_size)

        return self.W_o(attn_output)  # (batch_size, seq_len, d_model)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, dropout):
        super().__init__()

        self.lin1 = nn.Linear(in_dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, out_dim)

        self.dropout = dropout

    def forward(self, input):
        x = F.relu(self.lin1(input))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x  # (batch_size, seq_len, out_dim)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_atten = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(
            d_model, d_ff, d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, mask=None):
        attn_output = self.self_atten.forward(input, mask)
        x = self.norm1(input + self.dropout(attn_output))
        ff_output = self.feed_forward.forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x  # (batch_size, seq_len, d_model)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # model parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout

        # linear layers
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, input, batch_size) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, d_k)
        """
        return input.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def _scaled_dot_product_attention(self, Q, K, V, mask=None) -> torch.Tensor:
        """
        Compute the scaled dot product attention.
        """
        attention = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # mask attention
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -float('inf'))

        weights = F.softmax(attention.float(), dim=-1).type_as(attention)
        weights = F.dropout(weights, p=self.dropout, training=self.training)

        return torch.matmul(weights, V)

    def _combine_heads(self, input, batch_size) -> torch.Tensor:
        """
        Transpose and then concatenate the last two dimensions.
        """
        return input.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

    def forward(self, q, kv, mask=None) -> torch.Tensor:
        """
        Compute the multi-head attention.
        """
        batch_size, seq_len, d_model = q.size()

        assert d_model == self.d_model, "input tensor must have the same dim as d_model"

        Q = self._split_heads(self.W_q(q), batch_size)
        K = self._split_heads(self.W_k(kv), batch_size)
        V = self._split_heads(self.W_v(kv), batch_size)

        attn_output = self._scaled_dot_product_attention(Q, K, V, mask)
        attn_output = self._combine_heads(attn_output, batch_size)

        return self.W_o(attn_output)  # (batch_size, seq_len, d_model)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_atten = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.cross_atten = MultiHeadCrossAttention(
            d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(
            d_model, d_ff, d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        atten_output = self.self_atten(x, tgt_mask)
        x = self.norm1(x + self.dropout(atten_output))
        atten_output = self.cross_atten(x, enc_output, src_mask)
        x = self.norm2(x + self.dropout(atten_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(self, params):
        super().__init__()

        # embedding layers
        self.encoder_embedding = nn.Embedding(
            params.n_words,
            params.emb_dim,
            params.pad_index
        )
        self.decoder_embedding = nn.Embedding(
            params.n_words,
            params.emb_dim,
            params.pad_index
        )

        # positional encoding
        self.positional_encoding = PositionalEncoding(
            params.emb_dim,
            params.max_seq_len
        )

        # encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                params.emb_dim,
                params.n_heads,
                params.ffn_dim,
                params.dropout
            )
            for _ in range(params.n_enc_layers)
        ])

        # decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                params.emb_dim,
                params.n_heads,
                params.ffn_dim,
                params.dropout
            )
            for _ in range(params.n_dec_layers)
        ])

        # output layer
        self.output_layer = nn.Linear(params.emb_dim, params.n_words)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)

        seq_length = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        src_embedded = self.positional_encoding(self.encoder_embedding(src))
        tgt_embedded = self.positional_encoding(self.decoder_embedding(tgt))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.output_layer(dec_output)

        return output
