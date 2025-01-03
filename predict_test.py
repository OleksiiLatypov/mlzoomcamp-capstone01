import requests


url = "http://localhost:9696/predict"


# COVID IMAGES

data = {'url': 'https://www.ochsnerjournal.org/content/ochjnl/21/2/126/F7.large.jpg'}
#data = {'url': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTP4ycheoSrg-UHd-6sTSpv3w1FFGWkhYYMQ0bnwsS5XS-TN8LTm6pHd-0ukiuJ2DF1qmg&usqp=CAU'}

#NORMAL

#data = {'url': 'https://img.medscapestatic.com/pi/meds/ckb/59/16959tn.jpg'}

#PNEUMONIA IMAGES

#data = {'url': 'https://d2jx2rerrg6sh3.cloudfront.net/image-handler/picture/2020/6/shutterstock_786937069.jpg'}
#data = {'url': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcROcdcjNCSuEvy4awtsv_oC9f2jZm3y8gwnHAe4I9V_ynioqEV6nIia7A7aF2KNfONY3nk&usqp=CAU'}
#data = {'url': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQSuKQVUUP4GcDDqikhrFbZdlZMoM-xAm57kne6aICkA699oIXBlshK_R9z2DKQAR9wIFQ&usqp=CAU'}

result = requests.post(url, json=data).json()
print(result)