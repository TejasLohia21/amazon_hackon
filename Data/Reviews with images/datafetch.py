import requests

# URLs for Cell Phones and Accessories 5-core reviews and metadata
review_url = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Cell_Phones_and_Accessories_5.json.gz"
metadata_url = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/meta_Cell_Phones_and_Accessories.json.gz"

def download_file(url, filename):
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Finished downloading {filename}.")

if __name__ == "__main__":
    download_file(review_url, "Cell_Phones_and_Accessories_5.json.gz")
    download_file(metadata_url, "meta_Cell_Phones_and_Accessories.json.gz")
