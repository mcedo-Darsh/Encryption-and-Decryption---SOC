# NN Implementation
Well, we understood the maths behind the Neural Networks, just a simple matrix multiplication going on again and again. Just to summarize, If you have an input array of
some size say N, the output after the first layer is W matrix multiplied by with the input matrix with a bias matrix added. Then, a function, a non linear one in general is 
used to get the elements of the next layer

Step 1: Z(n) = W * N(n) + B 
Step 2: N(n+1) = f(Z(n))

Hence, any node is just a weighted sum of the previous nodes and some activation function associated with it.
brief summary again: https://www.youtube.com/watch?v=jmmW0F0biz0

To implement it using python:     
https://www.youtube.com/watch?v=Jy4wM2X21u0&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=3

My implementation of the NN on MNIST data:  
https://colab.research.google.com/drive/1oSHw0G9h9lmHAy7ZXa782bOa_zGNVWfd?usp=sharing

Go through the code, it might feel that it is a little bit ovewhelming, but with practice you would start to undersatand the foundations. Tis is the general syntax and yould be used further in many models
