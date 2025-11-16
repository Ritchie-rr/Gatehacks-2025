import torch
import torch.nn as nn
import config

class ASL_BiLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 222,                    # 222 features per frame
        hidden_size: int = config.HIDDEN_DIM,
        output_size: int = config.NUM_CLASSES,   # number of ASL gestures
        num_layers: int = config.LSTM_LAYERS,
        dropout: float = config.DROPOUT_RATE,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,               
            hidden_size=hidden_size,             
            num_layers=num_layers,               
            batch_first=True,                    
            dropout=dropout,
            bidirectional=True                   
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, 60, 222)
        """

        # LSTM output: (batch, seq_len, hidden_size*2)
        lstm_out, _ = self.lstm(x)

        # Take last time-step (frame 59)
        last_hidden = lstm_out[:, -1, :]

        # Classifier
        logits = self.fc(last_hidden)

        # # Convert to probabilities
        # probs = self.softmax(logits)

        return logits