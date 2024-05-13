
import requests
import pandas as pd
import json
from io import StringIO
import matplotlib



class PV_data():
    token = '168f2fc6d2e5e141d1d5c7bad57cd31e7aa30549'
    api_base = 'https://www.renewables.ninja/api/'
    default_value_cap =1

    def __init__(self, lat, long ,cap ,Hemisphere ,year):
        self.lat =lat
        self.long = long
        self.cap = cap if cap is not None else self.default_value_cap
        self.Hemisphere = Hemisphere
        self.year = year
        if self.Hemisphere == 'North':
            self.tilt = 1.3793 + self.lat * (1.2011 +self.lat * (-0.0144 + self.lat * 0.000080509))
            self.azimuth = 180
        else:
            self.tilt = -0.41657 + self.lat * (1.4216 + self.lat * (0.024051 + self.lat * 0.00021828))
            self.azimuth = 0

    def get_data(self):

        s = requests.session()
        # Send token header with each request
        s.headers = {'Authorization': 'Token ' + self.token}

        url = self.api_base + 'data/pv'

        args = {
            'lat': self.lat,
            'lon': self.long,
            'date_from': f'{self.year}-01-01',
            'date_to': f'{self.year}-12-31',
            'dataset': 'merra2',
            'capacity': 1.0,
            'system_loss': 0.1,
            'tracking': 0,
            'tilt': self.tilt,
            'azim': self.lat,
            'format': 'json'
        }

        r = s.get(url, params=args)

        # Parse JSON to get a pandas.DataFrame of data and dict of metadata
        parsed_response = json.loads(r.text)

        json_data = json.dumps(parsed_response['data'])
        data = pd.read_json(StringIO(json_data), orient='index')

        # data = pd.read_json(json.dumps(parsed_response['data']), orient='index')
        metadata = parsed_response['metadata']
        return data

#
# data = PV_data(lat=30,
#                        long=40,
#                        cap=None,
#                        Hemisphere='North',
#                        year=2019).get_data()
