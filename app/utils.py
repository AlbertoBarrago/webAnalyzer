import logging

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger("webAnalyzer")


def extract_text_from_url(url: str) -> str:
    """
    Extract text from url
    :param url:
    :return:
    """
    logger.info(f"Downloading content from URL: {url}")
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch URL: {url} (Status code: {response.status_code})")

    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()
