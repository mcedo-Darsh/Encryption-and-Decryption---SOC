# Vigenere Cipher

Vigenere cipher is a polyalphabetic cipher, unlike the monoalphabetic ones- substitution and ceiser (where the alphabet was mapped to a single alphabet- one to one function), but in vigenere cipher we do one to many mapping.
A key is used to create such one to many mapping, and main objective is to figureout the key while deciphering it.      

Vigenere cipher details:    
https://www.geeksforgeeks.org/dsa/vigenere-cipher/

# Deciphering Vigenere Cipher 
To decipher the vigenere cipher we need to find the key, but the key might be anything any word- hence for simplicity we assume the key might have a length less than or equal to 5, since the avg word length in English is 4.7 (approx. 5).
and then we have to find what the word is after deciphering its length. So for this we need to make two models- one, which can predict the len of the word, second, the word itself.

# Part 1
Making a model to find out the length of the Key

How to make it?
1. Make your dataset by importing text from any book, use this code to get one:         

import nltk        
nltk.download('gutenberg')        
from nltk.corpus import gutenberg               
text_net = (        
    gutenberg.raw('austen-emma.txt')+gutenberg.raw('austen-persuasion.txt')             
).lower()  

2. Choose your symbols set like done in ceiser cipher, all lower english letters +' '+ and other symbols. So make sure you remove other symbols and convert upper case to lower in the text.
3. Divide the data into 5 parts, and each part to some 100 letter wide blocks. Make a dataset function which generates random n length key and applies vigenere cipher to the text inputted, hence text,n are given as input it, ciphers it by using the Symbols set in step 2.
4. Now, pass on each part to the function and use the n= part number+1, i.e. 0th part->n=1, 1st part->n=2,etc. Also make a label set, it is simple- [0,0,0,..len(part[0])][1,1,1..len(part[1])].... [REMEMBER: we are not starting from 1 even if in the function we do part+1, because in cross entropy loss it expects us to start the labels from 0, dw we can add 1 at the end]
5. Make a LSTM class use nn.LSTM and make it bidirectional, (bidirectional=True) and since it is bidirectinal the input to fc is 2*hidden_size. The forward layer looks like this:  
   
    def forward(self, x):          
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirectional      
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)      
        out, _ = self.lstm(x, (h0, c0))      
        out = self.fc(out[:, -1, :])        
        return out      
7. make a dataset, where it converts the cipher letters into one hot encoding np.array and its corresponding label. (each block is of 100 ciphered letters, hence the one hot would be= 100 x len(symbols))
8. now train the model and test it
