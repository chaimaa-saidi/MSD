import requests

url = 'http://localhost:5000/predict' 

image_file_path = 'C:\\Users\\hp\\Desktop\\DataSet_MLOps\\chest_xray\\chest_xray\\test\\NORMAL\\IM-0011-0001-0002.jpeg'
image = open(image_file_path, 'rb')

files = {'image': ('image.jpg', image)}

response = requests.post(url, files=files)

print(response.json())
