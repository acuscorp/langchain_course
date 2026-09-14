import os
import pickle
from langchain_google_community import GoogleDriveLoader
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

credentials_path = "C:\\Users\\Noe.Acuna\\curso_langchain\\Tema 3\\credentials.json"
token_path = "C:\\Users\\Noe.Acuna\\curso_langchain\\Tema 3\\token.json"

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES) 
creds = flow.run_local_server(port=0) 
loader = GoogleDriveLoader(folder_id="1JuBNGBS7G-SDX2eiQ4FPfzCdMoFaQJPR",credentials=creds,recursive=True) 
documents = loader.load() 

print(documents)

