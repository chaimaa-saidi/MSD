import requests

url = 'http://localhost:5000/predict' 

image_file_path = 'C:\\Users\\hp\\MSD\\data\\processed\\Processed_TP1_Dataset\\test\NORMAL\\IM-0016-0001.jpeg'
image = open(image_file_path, 'rb')

files = {'image': ('image.jpg', image)}

response = requests.post(url, files=files)

print(response.json())
