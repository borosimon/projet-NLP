from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
import re
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import os
import time

os.environ["webdriver.chrome.driver"] = "/usr/lib/chromium-browser/chromedriver"

options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')

UVsdict={
    "auteur":[],
    'desc':[],
    'likes':[],
    'comments':[],
    "nb_comments":[],
    'shares':[],
}

driver = webdriver.Chrome(options=options)

driver.get('https://web.facebook.com/uvburkina')
driver.implicitly_wait(10)


last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(10)  # Attendre que le contenu se charge
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

html = driver.page_source
soup = bs(html, "html.parser")
publications = soup.find_all("div", class_="html-div xdj266r x11i5rnm xat24cr x1mh8g0r xexx8yu x4uap5 x18d9i69 xkhd6sd")
for publication in publications:
    texte_publication = publication.find_all("span",class_="x193iq5w xeuugli x13faqbe x1vvkbs x10flsy6 x1lliihq x1s928wv xhkezso x1gmr53x x1cpjm7i x1fgarty x1943h6x x4zkp8e x41vudc x6prxxf xvq8zen xo1l8bm xzsf02u x1yc453h")
    for texte_publication in texte_publication:
      desc=texte_publication.get_text()
      UVsdict['desc'].append(desc)

    date_publications = publication.find_all("span",class_="x1rg5ohu x6ikm8r x10wlt62 x16dsc37 xt0b8zv")
    for date_publication in date_publications:
      date=date_publication.get_text()
      UVsdict['auteur'].append(date)

    like_publications = publication.find_all("div",class_="x6s0dn4 x78zum5 x1iyjqo2 x6ikm8r x10wlt62")
    for like_publication in like_publications:
      like=like_publication.get_text()
      UVsdict['likes'].append(like)

    nb_comments_publications = publication.find_all("span",class_="html-span xdj266r x11i5rnm xat24cr x1mh8g0r xexx8yu x4uap5 x18d9i69 xkhd6sd x1hl2dhg x16tdsg8 x1vvkbs x1sur9pj xkrqix3")
    for comments_publication in nb_comments_publications:
      comments=comments_publication.get_text()
      UVsdict['nb_comments'].append(comments)

    share_publications = publication.find_all("span",class_="x193iq5w xeuugli x13faqbe x1vvkbs x10flsy6 x1lliihq x1s928wv xhkezso x1gmr53x x1cpjm7i x1fgarty x1943h6x x4zkp8e x41vudc x6prxxf xvq8zen xo1l8bm xi81zsa")
    for share_publication in share_publications:
      share=share_publication.get_text()
      UVsdict['shares'].append(share)

    comments_publications = publication.find_all("div",class_="xmjcpbm x1tlxs6b x1g8br2z x1gn5b1j x230xth x9f619 xzsf02u x1rg5ohu xdj266r x11i5rnm xat24cr x1mh8g0r x193iq5w x1mzt3pk x1n2onr6 xeaf4i8 x13faqbe")
    for comment_publication in comments_publications:
      comment=comment_publication.get_text()
      UVsdict['comments'].append(comment)

max_len = max(len(value) for value in UVsdict.values())
for key, value in UVsdict.items():
    if len(value) < max_len:
        UVsdict[key] = value + [None] * (max_len - len(value))

UVsd = pd.DataFrame(UVsdict)
UVsd.to_csv('UV_data_facebook.csv')
print(UVsdict)

driver.quit()