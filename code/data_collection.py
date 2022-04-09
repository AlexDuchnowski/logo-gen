import requests
from bs4 import BeautifulSoup

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

for name in company_names:
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