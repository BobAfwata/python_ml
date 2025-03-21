#!/usr/bin/python3

#strings :alphanumberic characters abc...z ABCD...Z , 0 1,2,34,...9 , special charactars

city = "seoul"
# uppercase / lower case

number = 78
print(city.upper())

name = "BOB"
print(type(name))

print(type(number))

#float : fractions / decimal point numbers 
weight = 78.9879

print(weight)
print("%f " %weight)
print("%.1f " %weight) # one decimal point
print("%.2f " %weight) # 2 decimal point

# using .format() to print floats 
print("{:.3f}" .format(weight))

#using fstring
print(f"{weight}")

#boolean

is_married = True
is_late = False

print(is_married)
print(type(is_married))