from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pandas as pd
import re

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']



def return_params_df(sheet_url, credentials, N=10):

    '''
    Name: return_params_df

    Purpose: The purpose of this function is to retieve the fixed and variable model parameters 
    from a Google Sheet. The access is performed via the Google Sheets API through a standard 
    OAuth protocol. 

    Args:
        sheet_url: The raw string URL of the google sheet you'd like to access.
        credentials: Path to the credentials.json file. 
        N: number of rows to read in

    Returns:
        A dataframe containing all the fixed parameters.
    '''
    
    ##PLEASE SAVE CREDENTIALS as <filepath>/credentials.json##
    
    #Creating token pickle file
    directory = credentials.split('credentials')[0]
    tokendir = directory + 'token.pickle'
    
    url = re.findall('/spreadsheets/d/([a-zA-Z0-9-_]+)', sheet_url)
    
    SAMPLE_SPREADSHEET_ID = url[0]
    creds = None
    
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time. This requires the USER to login and validate the google account from 
    # which the sheet is accessed from.

    if os.path.exists(tokendir):
        with open(tokendir, 'rb') as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials, SCOPES)
            creds = flow.run_local_server(port=0)
            
        # Save the credentials for the next run
        with open(tokendir, 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)

    # Call the Sheets API
    sheet = service.spreadsheets()
    

    result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID, range='FixedParams').execute()
    values = result.get('values', [])
    dataframe_sheet = pd.DataFrame(values)
    dataframe_sheet = dataframe_sheet.rename(columns=dataframe_sheet.iloc[0]).drop(dataframe_sheet.index[0])

    for col in dataframe_sheet.columns: 
        if col != 'model_id': 
            dataframe_sheet[col] = pd.to_numeric(dataframe_sheet[col])

    return dataframe_sheet.head(N)
    #return dict(zip(dataframe_sheet.columns.tolist(), dataframe_sheet[dataframe_sheet['model_id'] == modelname].values[0].tolist()))
    