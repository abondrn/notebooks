import requests
from time import sleep
import pandas as pd
import numpy as np
import os

GMAPS_API_KEY = 'AIzaSyAmqsSIKj7M9MNao_2xyKCj0UNcrMwlIx0'

if os.path.exists('GeocodedCities.csv'):
    cities = pd.read_csv('GeocodedCities.csv')
else:
    cities = pd.read_csv('Cities.csv')
    cities['Lat'] = np.NaN
    cities['Lng'] = np.NaN
    cities['TimeZone'] = np.NaN

LAT = cities.columns.get_loc('Lat')
LNG = cities.columns.get_loc('Lng')
TZ = cities.columns.get_loc('TimeZone')

for index, row in cities.iterrows():
    try:
        name = row['City']+(', '+row['State'] if not pd.isnull(row['State']) else '')
        if pd.isnull(cities.iloc[index,LAT]):
            data = requests.get('https://maps.googleapis.com/maps/api/geocode/json',
                                params={'address': name.replace(' ','+'), 'key': GMAPS_API_KEY}).json()
            lat = data['results'][0]['geometry']['location']['lat']
            lng = data['results'][0]['geometry']['location']['lng']
            sleep(1.5)
            cities.iloc[index,LAT] = lat
            cities.iloc[index,LNG] = lng
        if pd.isnull(cities.iloc[index,TZ]):
            data = requests.get('https://maps.googleapis.com/maps/api/timezone/json',
                                params={'location': str(lat)+','+str(lng), 'timestamp': 0, 'key': GMAPS_API_KEY}).json()
            sleep(1.5) # rate limiting
            cities.iloc[index,TZ] = data['rawOffset']/60/60
        print (name)
    except Exception as e:
        print (data)
        print (e)
        break

cities.to_csv('GeocodedCities.csv', index=False)