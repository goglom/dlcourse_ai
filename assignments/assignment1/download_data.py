import requests as re
import os

try:
    os.mkdir("data")
except FileExistsError:
    pass

train_url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
train_res = re.get(train_url)

with open("data/train_data.txt", "w") as data:
    data.write(train_res.text)
