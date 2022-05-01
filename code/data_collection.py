import requests
from bs4 import BeautifulSoup
import os

# creating a directory to save images
folder_name = 'images'
if not os.path.isdir(folder_name):
    os.makedirs(folder_name)

def download_image(url, folder_name, num):
    # write image to file
    response = requests.get(url)
    if response.status_code==200:
        with open(os.path.join(folder_name, str(num)+".jpg"), 'wb') as file:
            file.write(response.content)

company_names = [
    "Adidas",
    "Nike",
    "Apple",
    "Google",
    "McDonald's",
    "Microsoft",
    "BMW",
    "KFC",
    "Whole Foods",
    "American Airlines",
    "Delta Airlines"
]

url_base = "https://www.google.com/search?q="
image_specifier = "&tbm=isch"

image_links = []

for i in range(len(company_names)):
    name = company_names[i]
    print(name)
    url = url_base
    for word in name.split():
        url += word + "+"
    url += "logo"
    url += image_specifier
    sauce = requests.get(url)
    soup = BeautifulSoup(sauce.content,'html.parser')
    image = soup.find_all('img')[1]
    image_link = image['src']
    image_links.append(image_link)
    print(image_link)
    download_image(image_link, folder_name, i)