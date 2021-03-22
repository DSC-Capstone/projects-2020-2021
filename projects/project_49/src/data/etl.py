import time
import pandas as pd
import math

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException


def scrollDown(driver, n_scroll):
    body = driver.find_element_by_tag_name("body")
    while n_scroll >= 0:
        body.send_keys(Keys.PAGE_DOWN)
        n_scroll -= 1
    return driver

def scrollUp(driver, n_scroll):
    body = driver.find_element_by_tag_name("body")
    while n_scroll >= 0:
        body.send_keys(Keys.PAGE_UP)
        n_scroll -= 1
    return driver

def get_item_data(item_urls, driver):
    subset = item_urls
    start_index = 0
    for i in range(start_index, len(subset)):
        url = subset.URL[i]
        print(url)
        driver.get(url)
        time.sleep(5)
        browser = scrollDown(driver, 1)
        time.sleep(5)
        browser = scrollDown(driver, 1)
        time.sleep(5)

        # brand, name, price, ingredients, productID, subCategory, description
        subset.brand[i] = driver.find_element_by_class_name('css-57kn72').text
        subset.name[i] = driver.find_element_by_xpath('/html/body/div[1]/div[2]/div/main/div/div[1]/div/div[2]/div[1]/div[1]/h1/span').text
        subset.price[i] = driver.find_element_by_xpath('/html/body/div[1]/div[2]/div/main/div/div[1]/div/div[2]/div[1]/div[2]/div[1]/span').text
        subset.productID[i] = driver.find_element_by_xpath('/html/body/div[1]/div[2]/div/main/div/div[1]/div/div[2]/div[1]/div[1]/div[1]').text.split('ITEM ')[1]
        subset.subCategory[i] = driver.find_elements_by_xpath('/html/body/div[1]/div[2]/div/main/div/div[1]/nav/ol/li')[-1].text
        subset.description[i] = driver.find_element_by_class_name('css-pz80c5').text

        tab_items = driver.find_element_by_xpath('//div[@aria-label="Product Information"]')
        tab_buttons = tab_items.find_elements_by_xpath('//button[@data-comp="Link Box StyledComponent BaseComponent "]')
        ingredients_index = 0
        ingredients_found = False
        for button in tab_buttons:
    #         print(button.text)
            if (button.text == 'Details'):
                ingredients_index = 0
            if (button.text == "Ingredients"):
                ingredients_found = True
                break
            ingredients_index += 1
        if (ingredients_found):
            driver.find_element_by_id('tab' + str(ingredients_index)).click()
            ingredients_tab = driver.find_element_by_id('tabpanel' + str(ingredients_index))
            subset.ingredients[i] = ingredients_tab.find_element_by_class_name('css-pz80c5').text

        print(i)

        # counter += 1

def get_review_data(item_urls, driver, reviews_df):
    subset = item_urls
    for i in range(0, len(subset)):
        url = subset.URL[i]
        print(url)
        print(i)
        driver.get(url)
        productId = driver.find_element_by_class_name("css-1fuze8b.e65zztl0").text.split()[-1]
    #     print(productId)
    #     break
        browser = scrollDown(driver, 2)
        time.sleep(5)
        browser = scrollDown(driver, 2)
        time.sleep(5)
        browser = scrollDown(driver, 2)
        time.sleep(5)
        browser = scrollDown(driver, 2)
        time.sleep(5)
        browser = scrollUp(driver, 4)
        time.sleep(10)
        
        btn = driver.find_elements_by_id('review_filter_sort_trigger')
        tries = 0
        while ((len(btn) == 0) and (tries < 5)):
            url = subset.URL[i]
            driver.get(url)
            productId = driver.find_element_by_class_name('css-1fuze8b.e65zztl0').text.split()[-1]
            browser = scrollDown(driver, 2)
            time.sleep(5)
            browser = scrollDown(driver, 2)
            time.sleep(5)
            browser = scrollDown(driver, 2)
            time.sleep(5)
            browser = scrollDown(driver, 2)
            time.sleep(5)
            browser = scrollUp(driver, 4)
            time.sleep(10)
            btn = driver.find_elements_by_id('review_filter_sort_trigger')
            tries += 1
        if (tries == 5):
            print('some issue occured: product was probably not found')
            counter += 1
            continue
        btn = btn[0]
        btn.click()
        #sort by most helpful
        #btn = driver.find_elements_by_xpath('/html/body/div[1]/div[2]/div/main/div/div[2]/div[1]/div/div[3]/div/div/div[1]/div[2]/div[4]/div/div/div/div/div[1]/span')[0]
        btn = driver.find_elements_by_class_name('css-rfz1gg.eanm77i0')[0]
        btn.click()
        time.sleep(5)
        browser = scrollDown(driver, 3)
        time.sleep(5)
        #load more reviews
        try:
            for _ in range(360//12):
                btn = driver.find_elements_by_class_name('css-xswy5p.eanm77i0')
                if (len(btn) >= 1):
                    btn = btn[0]
                    btn.click()
                browser = scrollDown(driver, 4)
                time.sleep(5)
                btn = driver.find_elements_by_class_name('css-xswy5p.eanm77i0')
                if (len(btn) >= 1):
                    btn = btn[0]
                    btn.click()
        except NoSuchElementException:
            print("No more reviews to load!")
            
        reviews = driver.find_elements_by_class_name('css-13o7eu2.eanm77i0')
        for review in reviews:
            try:
                userId = review.find_elements_by_class_name('css-lmu1yj.eanm77i0')[0].text
            except IndexError:
                userId = ""
            rating = review.find_elements_by_class_name('css-4qxrld')[0].get_attribute("aria-label").split()[0]
                
            info = review.find_elements_by_class_name('css-74woun.eanm77i0')
            user_info = {'Skin Type': "", 'Skin Tone': ""}
            for attr in info:
                if 'Skin Type' in attr.text:
                    user_info['Skin Type'] = attr.text[10:]
                elif 'Skin Tone' in attr.text:
                    user_info['Skin Tone'] = attr.text[10:]
        
        # transform into a data frame
            dic = {'userID': userId, 'productID': productId, 'rating': rating, 'skin_tone': user_info['Skin Tone'], 'skin_type': user_info['Skin Type']}
            reviews_df = reviews_df.append(pd.DataFrame(dic, index=[0]), ignore_index = True)
    return reviews_df


def scrape_item_data(outdir, chrome_path, item_url_csv):
    driver = webdriver.Chrome(executable_path = chrome_path)
    item_urls = pd.read_csv(item_url_csv)
    df2 = pd.DataFrame(columns=['productID', 'brand', 'name', 'price', 'description', 'ingredients', 'subCategory'])
    item_urls = pd.concat([item_urls, df2], axis = 1)
    get_item_data(item_urls, driver)
    item_urls.to_csv(outdir + 'test_itemdata.csv', encoding = 'utf-8-sig', index = False)


def scrape_review_data(outidr, chrome_path, item_url_csv):
    driver = webdriver.Chrome(executable_path = chrome_path)
    item_urls = pd.read_csv(item_url_csv)
    reviews_df = pd.DataFrame(columns=['userID', 'productID', 'rating', 'skin_tone', 'skin_type'])
    get_review_data(item_urls, driver, reviews_df)
    reviews_df.to_csv(outdir + 'test_reviewdata.csv', encoding = 'utf-8-sig', index = False)