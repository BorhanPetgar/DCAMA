"""
You can download files from google drive with gdown!
you may have urls in these formats:
https://drive.google.com/file/d/1AZcc77cmDfkWA8f8cs-j-CUuFFQ7tPoK/view
https://drive.google.com/file/d/1fcwqp0eQ_Ngf-8ZE73EsHKP8ZLfORdWR/view
https://drive.usercontent.google.com/download?id=1AZcc77cmDfkWA8f8cs-j-CUuFFQ7tPoK&export=download&authuser=0
Then use the id within the urls to convert it in this format
https://drive.google.com/uc?id=1AZcc77cmDfkWA8f8cs-j-CUuFFQ7tPoK
"""
import gdown
# https://drive.google.com/drive/folders/1qgdD-D3CgyDa4SBp9a-5H4-rOr0vF5Fv
# https://drive.google.com/drive/folders/1qgdD-D3CgyDa4SBp9a-5H4-rOr0vF5Fv
# url = 'https://drive.google.com/uc?id=16IJeYqt9oHbqnSI9m2nTXcxQWNXCfiGb'
# output = 'val2014.zip'
# gdown.download(url, output, quiet=False)

# OPTION 2: Download an entire folder
folder_id = "1qgdD-D3CgyDa4SBp9a-5H4-rOr0vF5Fv"  # This is the ID from your commented URL
output_folder = "checkpoints"  # Where to save the folder contents

# Download the entire folder
gdown.download_folder(id=folder_id, output=output_folder, quiet=False)