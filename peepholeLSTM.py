import torch
import torch.nn as nn
import math


# peepholeLSTM.py
class PeepholeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, critical_init=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Add classification head
        self.classifier = nn.Linear(hidden_size, 10)  # 10 classes for MNIST

        # Input gate
        self.W_i = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.U_i = nn.Parameter(torch.empty(hidden_size, input_size))
        self.b_i = nn.Parameter(torch.empty(hidden_size))

        # Forget gate
        self.W_f = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.U_f = nn.Parameter(torch.empty(hidden_size, input_size))
        self.b_f = nn.Parameter(torch.empty(hidden_size))

        # Output gate
        self.W_o = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.U_o = nn.Parameter(torch.empty(hidden_size, input_size))
        self.b_o = nn.Parameter(torch.empty(hidden_size))

        # Cell state
        self.W_c = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.U_c = nn.Parameter(torch.empty(hidden_size, input_size))
        self.b_c = nn.Parameter(torch.empty(hidden_size))

        self._reset_parameters(critical_init)

    def _reset_parameters(self, critical_init):
        """Initialize parameters either with critical or standard initialization"""
        if critical_init:
            # Critical initialization as per paper
            # μᵢ, μᵣ, μₒ, ρ²ᵢ, ρ²ᶠ, ρ²ᵣ, ρ²ₒ, ν²ᵢ, ν²ᶠ, ν²ᵣ, ν²ₒ = 0
            # μᶠ = 5
            # σ²ᵢ, σ²ᶠ, σ²ᵣ, σ²ₒ = 10⁻⁵
            std = math.sqrt(1e-5 / self.hidden_size)

            # Weight initializations
            for W in [self.W_i, self.W_f, self.W_o, self.W_c]:
                nn.init.normal_(W, std=std)

            for U in [self.U_i, self.U_f, self.U_o, self.U_c]:
                nn.init.normal_(U, std=std)

            # Bias initializations
            nn.init.zeros_(self.b_i)  # μᵢ = 0
            nn.init.constant_(self.b_f, 5.0)  # μᶠ = 5
            nn.init.zeros_(self.b_o)  # μₒ = 0
            nn.init.zeros_(self.b_c)

        else:
            # Standard initialization (Xavier/Glorot)
            for W in [self.W_i, self.W_f, self.W_o, self.W_c]:
                nn.init.xavier_uniform_(W)

            for U in [self.U_i, self.U_f, self.U_o, self.U_c]:
                nn.init.xavier_uniform_(U)

            # Initialize biases to zeros except forget gate
            nn.init.zeros_(self.b_i)
            nn.init.ones_(self.b_f)  # Common practice for LSTM
            nn.init.zeros_(self.b_o)
            nn.init.zeros_(self.b_c)

        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x, init_states=None):
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()

        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states

        hidden_seq = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)

            # Gates with peephole connections
            i_t = torch.sigmoid(
                h_t @ self.W_i.T +  # (batch_size, hidden_size)
                x_t @ self.U_i.T +  # (batch_size, hidden_size)
                self.b_i  # (hidden_size)
            )

            f_t = torch.sigmoid(
                h_t @ self.W_f.T +
                x_t @ self.U_f.T +
                self.b_f
            )

            o_t = torch.sigmoid(
                h_t @ self.W_o.T +
                x_t @ self.U_o.T +
                self.b_o
            )

            # Cell state
            c_tilde = torch.tanh(
                h_t @ self.W_c.T +
                x_t @ self.U_c.T +
                self.b_c
            )

            c_t = f_t * c_t + i_t * c_tilde
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t)

        hidden_seq = torch.stack(hidden_seq, dim=1)
        return hidden_seq, (h_t, c_t)