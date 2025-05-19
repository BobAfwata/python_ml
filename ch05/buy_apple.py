#A python program to calculate the prices of apples using forward aand backward propergation
#with naive_layer module
#Bob Afwata <bafwata@gmail.com>
# 24/3/2025
#  Import the naive layer module
from layer_naive import *


apple = 150
apple_num = 2
tax = 1.1

#instantiate the Multiplication layer from the module imported.
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# perform forward propagation
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

# perform backward propagation
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)


#print the prices for apple and Tax
print("price:", int(price))
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))
print("dTax:", dtax)
