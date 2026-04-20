import csv
import requests
import pandas as pd
from tqdm import tqdm
from io import StringIO

from trade.constants import DOWNLOAD_FILE


def download_searching_space(url_file, delimiter='\t'):
    """
    Download txt files from URLs in a CSV file and convert them to CSV files.

    :param url_file: Path to the CSV file containing URLs.
    :param delimiter: Delimiter used in the downloaded txt files (default is tab '\t').
    """

    with open(url_file, newline='') as file:
        reader = file.readlines()
        for url in tqdm(reader):
            url = url.strip()
            try:
                response = requests.get(url)
                response.raise_for_status()

                file_name = DOWNLOAD_FILE / url.split('/')[-1].replace('.txt', '.csv')
                txt_data = StringIO(response.text)
                data = pd.read_csv(txt_data, delimiter=delimiter, engine='python')
                data.to_csv(file_name, index=False)

            except requests.exceptions.RequestException as e:
                print(f"Failed to download {url}: {e}")
            except pd.errors.ParserError as e:
                print(f"Failed to parse {url}: {e}")


if __name__ == '__main__':
    from tap import tapify
    tapify(download_searching_space)
