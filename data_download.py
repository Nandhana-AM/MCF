"""
Utility for downloading images from URLs
"""
import os
import urllib.request
import multiprocessing
from pathlib import Path
from functools import partial
from tqdm import tqdm


def download_image(image_link, savefolder):
    """
    Download a single image from URL
    
    Args:
        image_link: URL of the image
        savefolder: Directory to save the image
    """
    if isinstance(image_link, str):
        filename = Path(image_link).name
        image_save_path = os.path.join(savefolder, filename)
        
        if not os.path.exists(image_save_path):
            try:
                urllib.request.urlretrieve(image_link, image_save_path)
            except Exception as ex:
                print(f'Warning: Not able to download - {image_link}\n{ex}')
        else:
            return
    return


def download_images(image_links, download_folder, num_workers=100):
    """
    Download multiple images in parallel
    
    Args:
        image_links: List of image URLs
        download_folder: Directory to save images
        num_workers: Number of parallel workers
    """
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    
    results = []
    download_image_partial = partial(download_image, savefolder=download_folder)
    
    with multiprocessing.Pool(num_workers) as pool:
        for result in tqdm(pool.imap(download_image_partial, image_links), 
                          total=len(image_links), 
                          desc=f"Downloading to {download_folder}"):
            results.append(result)
        pool.close()
        pool.join()
    
    print(f"Downloaded images to {download_folder}")