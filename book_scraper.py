from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


def scrape_books(query):

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    books = []

    try:
        search_url = f"https://books.google.com/books?q={query}"
        driver.get(search_url)

        wait = WebDriverWait(driver, 10)
        wait.until(
            EC.presence_of_element_located((By.XPATH, "//h3[@class='gs_rt']/a"))
        )

        results = driver.find_elements(By.XPATH, "//h3[@class='gs_rt']/a")

        for r in results:
            title = r.text.strip()
            link = r.get_attribute("href")

            if title and link and "preview" not in title.lower():
                books.append({
                    "title": title,
                    "link": link
                })

            if len(books) == 5:
                break

        driver.quit()
        return books

    except Exception as e:
        print("Scraping error:", e)
        driver.quit()
        return []