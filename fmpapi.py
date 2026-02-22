# Get market data from fmp (financialmodelingprep.com) and save it to a csv file each n min or n hours
# https://financialmodelingprep.com/api/v3/technical_indicator/30min/AAPL?type=standardDeviation&period=10&apikey=xQZFfDNtJjyxghjNX7YPW4VaZO1WzTif
# Usage: python get_fmp_by_mh.py --symbol AAPL --start 2020-01-01 --end 2021-01-01 --output data.csv

import requests
import pandas as pd
import argparse
import pdb

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Get market data from fmp (financialmodelingprep.com)')
    parser.add_argument('--symbol', type=str, default='TSLA', help='Stock symbol to get data for')
    parser.add_argument('--interval', type=str, default='30min', help='Interval for data (1min, 5min, 30min, 1hour, 4hour)')
    parser.add_argument('--output', type=str, default='AAPL_30min.xlsx', help='Output Excel file name')
    return parser.parse_args()

'''Get market data from fmp (financialmodelingprep.com) and return a pandas DataFrame with : 
- open
- high
- low
- close
- volume
- date
:param symbol: Stock symbol to get data for
:param interval: Interval for data (1min, 5min, 30min, 1hour, 4hour)
'''
def get_fmp_data(symbol, interval='30min', APIKEY='TxlfXZeoBpDIPHnFcoWD8pv3KQ3zuJ5V', linear_extension=False):
    if interval == '1d':
        interval = '1day'
    url = f'https://financialmodelingprep.com/api/v3/technical_indicator/{interval}/{symbol}?type=standardDeviation&period=10&apikey={APIKEY}'
    #url = f'https://financialmodelingprep.com/api/v3/historical-chart/{interval}&symbol={symbol}&apikey={APIKEY}'
    #url = f'https://financialmodelingprep.com/stable/historical-chart/{interval}?symbol={symbol}&apikey={APIKEY}'
    print(f'Getting data from {url}')
    response = requests.get(url)
    data = response.json()
    # extraction des données open, high, low, close, volume, date de la liste des dictionnaires dans un dataframe
    data2 = []
    # Boucle
    for d in data:
        # Extract the date and the values
        date = d['date']
        open_ = d['open']
        high = d['high']
        low = d['low']
        close = d['close']
        volume = d['volume']
        # Append to data2
        data2.append({'date': date, 'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume})
    # Convert data to pandas dataframe
    pdata = pd.DataFrame(data2)
    pdata = pdata.sort_values('date')
    # reset index
    pdata.reset_index(drop=True, inplace=True)
    if linear_extension:
        # add 15 rows to the dataframe with the last values of open, close, high, low, volume   and date
        last_row = pdata.iloc[-1]
        for i in range(linear_extension):
            new_row = last_row.copy()
            new_row['date'] = pd.to_datetime(new_row['date']) + pd.Timedelta(minutes=5 * (i + 1))
            # Convert date to string format
            new_row['date'] = new_row['date'].strftime('%Y-%m-%d %H:%M:%S')
            # Add new_row to dataframe pdata
            pdata = pd.concat([pdata, pd.DataFrame([new_row])], ignore_index=True)

    return pdata

''' Get historical intraday data for a stock symbol from fmp (financialmodelingprep.com)
'''
def get_historical_intraday_data(symbol, start_date, end_date, interval='30min', APIKEY='TxlfXZeoBpDIPHnFcoWD8pv3KQ3zuJ5V'):
    # Get historical data from fmp
    url = f'https://financialmodelingprep.com/api/v3/historical-chart/{interval}/{symbol}?from={start_date}&to={end_date}&apikey={APIKEY}'
    print(f'Getting historical data from {url}')
    response = requests.get(url)
    data = response.json()
    # extraction des données open, high, low, close, volume, date de la liste des dictionnaires dans un dataframe
    data2 = []
    # Boucle
    for d in data:
        # Extract the date and the values
        date = d['date']
        open_ = d['open']
        high = d['high']
        low = d['low']
        close = d['close']
        volume = d['volume']
        # Append to data2
        data2.append({'date': date, 'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume})
    # Convert data to pandas dataframe
    pdata = pd.DataFrame(data2)
    pdata = pdata.sort_values('date')
    # reset index
    pdata.reset_index(drop=True, inplace=True)
    return pdata

''' Get historical intraday data for a stock symbol from fmp (financialmodelingprep.com) up to a specific date
'''
def get_historical_intraday_data_up_to(symbol, end_date, interval='30min', APIKEY='TxlfXZeoBpDIPHnFcoWD8pv3KQ3zuJ5V'):
    # Get historical data from fmp
    url = f'https://financialmodelingprep.com/api/v3/historical-chart/{interval}/{symbol}?to={end_date}&extended=true&apikey={APIKEY}'
    print(f'Getting historical data from {url}')
    response = requests.get(url)
    data = response.json()
    # extraction des données open, high, low, close, volume, date de la liste des dictionnaires dans un dataframe
    data2 = []
    # Boucle
    for d in data:
        # Extract the date and the values
        date = d['date']
        open_ = d['open']
        high = d['high']
        low = d['low']
        close = d['close']
        volume = d['volume']
        # Append to data2
        data2.append({'date': date, 'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume})
    # Convert data to pandas dataframe
    pdata = pd.DataFrame(data2)
    pdata = pdata.sort_values('date')
    # reset index
    pdata.reset_index(drop=True, inplace=True)
    return pdata
''' Get historical intraday data for a stock symbol from fmp (financialmodelingprep.com) from a specific date
'''
def get_historical_intraday_data_from(symbol, start_date, interval='30min', APIKEY='TxlfXZeoBpDIPHnFcoWD8pv3KQ3zuJ5V'):
    # Get historical data from fmp
    url = f'https://financialmodelingprep.com/api/v3/historical-chart/{interval}/{symbol}?from={start_date}&extended=true&apikey={APIKEY}'
    print(f'Getting historical data from {url}')
    response = requests.get(url)
    data = response.json()
    # extraction des données open, high, low, close, volume, date de la liste des dictionnaires dans un dataframe
    data2 = []
    # Boucle
    for d in data:
        # Extract the date and the values
        date = d['date']
        open_ = d['open']
        high = d['high']
        low = d['low']
        close = d['close']
        volume = d['volume']
        # Append to data2
        data2.append({'date': date, 'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume})
    # Convert data to pandas dataframe
    pdata = pd.DataFrame(data2)
    pdata = pdata.sort_values('date')
    # reset index
    pdata.reset_index(drop=True, inplace=True)
    return pdata
# get n values from fmp for AAPL beginning at a given date
def get_n_values_from_date(symbol, start_date, n=1000, interval='30min', APIKEY='TxlfXZeoBpDIPHnFcoWD8pv3KQ3zuJ5V'):
    ''' Get n values from fmp for a given symbol starting from a given date
    '''
    # split in chunks of 100 values
    all_data = pd.DataFrame()
    start = start_date
    end = pd.to_datetime(start_date) + pd.Timedelta(days=5)
    # convert to string
    #end = end.strftime('%Y-%m-%d %H:%M:%S')
    end = end.strftime('%Y-%m-%d')
    while True:
        if len(all_data)>= n:
            break

        data = get_historical_intraday_data(symbol, start, end, interval=interval, APIKEY=APIKEY)
        # concatenate data to all_data
        if data.empty:
            break
        data = data.sort_values('date')
        data.reset_index(drop=True, inplace=True)
        all_data = pd.concat([all_data, data], ignore_index=True)
        # remove duplicates in all_data
        all_data = all_data.drop_duplicates(subset=['date'])
        # sort all_data by date
        all_data = all_data.sort_values('date')
        all_data.reset_index(drop=True, inplace=True)
        tmpstart = all_data['date'].iloc[-1]
        # test si tmpstart et start sont égaux (yyyy-mm-dd) même jour
        if pd.to_datetime(tmpstart).strftime('%Y-%m-%d') == pd.to_datetime(start).strftime('%Y-%m-%d'):
            break
        else:
           start = tmpstart
        end = pd.to_datetime(start) + pd.Timedelta(days=5)
        # convert to string and keep both date and time 
        end = end.strftime('%Y-%m-%d')
    # keep only the first n values
    all_data = all_data.head(n)
    return all_data


#https://financialmodelingprep.com/api/v3/historical-chart/30min/AAPL?from=2020-08-01&to=2020-09-01&apikey=TxlfXZeoBpDIPHnFcoWD8pv3KQ3zuJ5V

def main():
    ''' Get market data from fmp (financialmodelingprep.com) and save it to a excel file 
    Usage: python fmpapi.py --symbol AAPL --interval 30min --output data.xlsx
    '''
    args = parse_args()
    symbol = args.symbol
    output_file = args.output
    pdata = get_fmp_data(symbol, interval='5min')
    print(pdata.tail())
    print(f'Longueur du dataframe : {len(pdata)}')
    # Save data to a excel ('xls') file using xlsxwriter
    # get historical data from fmp from 01-01-2015 to 01-02-2015 at 30min interval
    start_date = '2015-01-01'
    end_date = '2015-03-01'
    pdata = get_historical_intraday_data(symbol, start_date, end_date, interval='30min')
    # Test 1000 values from a given date
    # start_date = '2015-03-01 00:00:00'
    pdata = get_n_values_from_date(symbol, start_date, n=1000, interval='30min')
    print(pdata.head())
    print(f'Longueur du dataframe : {len(pdata)}')
    pdata.to_excel(output_file, index=False)
    print(f'Data saved to {output_file}')



if __name__ == '__main__':
    main()

