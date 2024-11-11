from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

# Set up the WebDriver (this example uses Chrome)

def getEbayImageLinks(cardName, numperpage):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode if you don't need a visible browser window
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    #p2334524.m570.l1313
    url = f"https://www.ebay.com/sch/i.html?_from=R40&_trksid=p2334524.m570.l1313&_nkw={cardName}+{numperpage}&_sacat=0"
    print(url)
    driver.get(url)
    # Wait for images to load (adjust time as needed)
    time.sleep(5)
    # Find all image elements and filter for .webp URLs
    images = driver.find_elements(By.TAG_NAME, "img")
    webp_links = [img.get_attribute("src") for img in images if img.get_attribute("src") and img.get_attribute("src").endswith(".webp")]
    # Output the links
    for link in webp_links:
        print(link)
    # Close the browser
    driver.quit()
    return webp_links

