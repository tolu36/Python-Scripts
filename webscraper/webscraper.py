# %%
from bs4 import BeautifulSoup

# with open('index.html', 'r') as f:
#     doc = BeautifulSoup(f, 'html.parser')

# print(doc.prettify())

# tags = doc.find_all('p')
# tags[0].find_all('b')
# %%
import requests

# url = 'https://www.newegg.ca/gigabyte-geforce-rtx-3070-ti-gv-n307tgaming-oc-8g/p/N82E16814932443?Item=N82E16814932443'

# results = requests.get(url)
# doc = BeautifulSoup(results.text, 'html.parser')

# tags = doc.find_all(text = '$')
# parent=tags[0].parent
# strong = parent.find('strong')
# print(strong.string)

# %%
url = "https://dutchie.com/embedded-menu/the-hunny-pot-cannabis-co1/product/hunny-pot-cannabis-limited-drip-3-5g"

results = requests.get(url)
doc = BeautifulSoup(results.text, "html.parser")
# %%
import os

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By

os.environ["PATH"] += r"./chromedriver_win32/chromedriver.exe"
driver = webdriver.Chrome()
driver.get(
    "https://dutchie.com/embedded-menu/the-hunny-pot-cannabis-co1/products/flower"
)
driver.implicitly_wait(30)
my_element = driver.find_elements(By.CSS_SELECTOR, '[data-testid="product-list-item"]')
html = my_element[0].get_attribute("innerHTML")
doc = BeautifulSoup(html, "html.parser")
# %%
