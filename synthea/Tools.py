import asyncio
import aiohttp
import trafilatura

from bs4 import BeautifulSoup

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.3'}
SCRAPING_TIMEOUT = 10
async def google_search_and_scrape(query: str) -> list:
    """
    Performs a Google search for the given query, retrieves the top search result URLs,
    and scrapes the text content from those pages

    Args:
        query (str): The search query.
    Returns:
        list: A list of dictionaries containing the URL, text content for each scraped page.
    """
    num_results = 2
    url = 'https://www.google.com/search'
    params = {'q': query, 'num': num_results}
    
    print(f"Performing google search with query: {query}...")
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers=HEADERS, timeout=SCRAPING_TIMEOUT) as response:
            html = await response.text()

    soup = BeautifulSoup(html, 'html.parser')
    urls = [result.find('a')['href'] for result in soup.find_all('div', class_='tF2Cxc')]
    
    print(f"Scraping text from urls, please wait...") 
    [print(url) for url in urls]

    if not urls:
        print("No search results found.")
        return []

    tasks = [asyncio.create_task(scrape_url(url)) for url in urls[:num_results]]
    results = await asyncio.gather(*tasks)

    return [result for result in results if result is not None]

async def scrape_url(url: str):
    """
    Downloads and scrapes a single page from the internet.

    Args:
        query (str): The search query.
    Returns:
        list: A list of dictionaries containing the URL, text content for each scraped page.
    """
    try:           
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=HEADERS, timeout=SCRAPING_TIMEOUT) as response:
                article_html = await response.text()

        # Extract main content using trafilatura
        text_content = trafilatura.extract(article_html)
        print(f"Finished scraping {url}")

        return {'url': url, 'content': text_content}
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None