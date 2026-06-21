# Finding the key using LSTM (OPTIONAL)
This segment is optional since it requires a Dual LSTM model, we can do this question using Transformers which would be more robust, so if you wat to do this using LSTM you can take this-

In the last part we found out the size of the key, This timw we will be finding the key itself, The LSTM now learns the repetition and the subsequennt repeating patter and would try to guess the key.     
You can follow these guidelines to make the LSTM:    
(We assume that the length of the key is 5 since, if it works for 5 it will work for the smaller lengths aswell)
1. Use the same dataset, the same class to cipher the text and the same dataloader
2. choose your symbols- A-Z,' ',...etc, and make a set ,say symbols. Now pre process the data in the same fashion as done in all the previous deciphering models. 
3. Instead of dividing the set into parts of different lengths, just divide the text into many segments of 100- 150 chr each, and now make the txt ciphered.
4. divide into test and train, and one hot encode the characters in each segment, so you would get a vector of size- 100xlen(symbols),Along with this also concatenate the Actual sentence so that your output becomes - 100x(len(symbols)*2).
5. Make a LSTM model- the model is a little bit different it has one encoder and a decoder branch, dw both are lstms only- th output of one will be fed to the other guy. A sample code is given:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Bidirectional_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x):
        outputs, (hn, cn) = self.lstm(x)
        
        hn_combined = torch.cat((hn[0, :, :], hn[1, :, :]), dim=1).unsqueeze(0)
        cn_combined = torch.cat((cn[0, :, :], cn[1, :, :]), dim=1).unsqueeze(0)
        
        return outputs, hn_combined, cn_combined

class Attention_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state, cell_state, encoder_outputs):
        hidden_reshaped = hidden_state.permute(1, 2, 0)
        
        raw_scores = torch.bmm(encoder_outputs, hidden_reshaped)
        attention_weights = F.softmax(raw_scores, dim=1)
        
        weights_flipped = attention_weights.permute(0, 2, 1)
        context = torch.bmm(weights_flipped, encoder_outputs)
        
        lstm_input = torch.cat((x, context), dim=2)
        
        out, (new_hidden, new_cell) = self.lstm(lstm_input, (hidden_state, cell_state))
        prediction = self.fc(out)
        
        return prediction, new_hidden, new_cell
```
6. Make the dataset and finally train the model and Test it
