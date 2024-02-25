from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from os.path import basename
from datetime import datetime

# Function to upload file to the drive
def upload_file(drive, file_path, folder_id):
    # Check if the file already exists in the folder
    file_list = drive.ListFile({'q': "'%s' in parents and title='%s'" % (folder_id, os.path.basename(file_path))}).GetList()
    if file_list:
        print(f"File '{os.path.basename(file_path)}' already exists in the Google Drive folder. Skipping upload.")
        return

    # Upload the file if it doesn't exist
    file_drive = drive.CreateFile({'title': os.path.basename(file_path), 'parents': [{'id': folder_id}]})
    file_drive.SetContentFile(file_path)  # Set the content of the file
    file_drive.Upload()
    print(f"{file_path} uploaded successfully to Google Drive.")

# Function to create a folder in Google Drive
def create_folder_drive(drive, folder_name, parent_folder_id=None):
    # Check if the folder already exists, if not create a new one
    folder_list = drive.ListFile({'q': "title='%s' and mimeType='application/vnd.google-apps.folder'" % folder_name}).GetList()
    if folder_list:
        return folder_list[0]['id']  # Return the existing folder ID
        
    folder_drive = drive.CreateFile({'title': folder_name, 'mimeType': 'application/vnd.google-apps.folder'})
    if parent_folder_id:
        folder_drive['parents'] = [{'id': parent_folder_id}]
    folder_drive.Upload()
    return folder_drive['id']

# Function to upload files to Google Drive   
def save_files_to_drive(drive, pi_folder, create_drive_folder, drive_folder_id):
    # Create a folder inside the folder drive to store the data
    folder_id = create_folder_drive(drive, create_drive_folder, drive_folder_id)

    # Iterate through files inside the Pi folder
    for filename in os.listdir(pi_folder):
        file_path = os.path.join(pi_folder, filename)
        
        # Check if the file exists in the Pi before uploading
        if os.path.isfile(file_path):
            upload_file(drive, file_path, folder_id)
        else:
            print(f"Error: File {file_path} not found!")
