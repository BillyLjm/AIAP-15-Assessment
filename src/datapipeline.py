import numpy as np
import pandas as pd
import sqlite3
import Levenshtein

class Datapipeline():

    def remove_dupl(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes rows with duplicate 'Ext_Intcode' from the cruise survey results
        The rows with the most non-NA values will be kept

        :param df: Pandas dataframe to remove duplicates from
        :return: Pandas dataframe with duplicate rows removed
        """
        df['counts'] = df.count(axis=1)
        df = df.sort_values(['Ext_Intcode', 'counts'])
        df = df.drop_duplicates('Ext_Intcode', keep='last')
        df = df.drop(columns=['index', 'counts'])
        df = df.set_index('Ext_Intcode')
        return df

    def clean_5star_rating(self, rating: str) -> int:
        """
        Converts a rating of 'Not at all important', 'A little important', 
        'Somewhat important', 'Very important', 'Extremely important' into a 
        ordinal 5-point scale

        :param rating: Pandas series to be converted
        :return: Pandas series converted to 5-point scale
        """
        scale_pre = {'Not at all important': 1, 'A little important': 2, 
                     'Somewhat important': 3, 'Very important': 4, 
                     'Extremely important': 5}
        if not rating in scale_pre: return rating 
        else: return scale_pre[rating]

    def clean_distance(self, dist: str) -> int:
        """
        Converts a distance specified as "<number> <units>" (e.g. "50 Miles")
        into the number of kilometres

        :param dist: distance specified as "<number> <units>"
        :return: number of kilometres the distance corresponds to
        """
        if pd.isnull(dist): return dist # null values
        num, units = dist.split(' ')
        num = int(num)
        if (num <= 0): num = np.nan
        elif (units == "km" or units == "KM"): num *= 1
        elif (units == "Miles"): num *= 1.60934
        return num

    def clean_name(self, name: str, valid_names: list) -> str:
        """
        Maps a name onto a finite set of valid names, based on Levenshtein 
        distance

        :param name: name to map onto `valid_names`
        :param valid_names: list of names that `name` should map onto
        :return: the element of `valid_names` that is closest to `name`
        """
        valid_names = ['Blastoise', 'Lapras']
        if pd.isnull(name): return name # null value
        elif name in valid_names: return name # valid name
        else:
            dist = [Levenshtein.distance(name, i) for i in valid_names]
            return valid_names[dist.index(min(dist))]


    def read_data(self, pre_path: str, post_path: str) -> pd.DataFrame:
        """
        Reads the pre-purchase and post-trip databases into one combined, 
        cleaned and comprehensive pandas DataFrame

        :param pre_path: filepath to the pre-purchase survey result database
        :param post_path: filepath to the post-trip survey result database
        :return: Pandas dataframe of the two databases combined and cleaned
        """
        # read pre data
        con_pre = sqlite3.connect(pre_path)
        df_pre = pd.read_sql('SELECT * FROM cruise_pre', con_pre)
        con_pre.close()

        # read post_data
        con_post = sqlite3.connect(post_path)
        df_post = pd.read_sql('SELECT * FROM cruise_post', con_post)
        con_post.close()

        # combine data
        df = self.remove_dupl(df_pre)
        df = df.join(self.remove_dupl(df_post))

        # convert 5-star rating to ordinal
        df['Onboard Wifi Service'] = df['Onboard Wifi Service'].apply(self.clean_5star_rating)
        df['Onboard Dining Service'] = df['Onboard Dining Service'].apply(self.clean_5star_rating)
        df['Onboard Entertainment'] = df['Onboard Entertainment'].apply(self.clean_5star_rating)

        # clean distance
        df['Cruise Distance'] = df['Cruise Distance'].apply(self.clean_distance)

        # clean names
        df['Cruise Name'] = df['Cruise Name'].apply(
            lambda x: self.clean_name(x, ['Blastoise', 'Lapras']))

        # clean datetime columns
        df['Date of Birth'] = pd.to_datetime(df['Date of Birth'], format='mixed')
        df['Date of Birth'] = df['Date of Birth'].apply(
            lambda x: x if x.year > 1920 else pd.NaT)

        df['Logging'] = pd.to_datetime(df['Logging'], format='mixed')
        df['Logging'] = df['Logging'].apply(
            lambda x: x if x.month < 9 and x.year <= 2023 else pd.NaT)

        # choose datatypes
        df = df.astype({
            'Gender': 'category', 
            'Date of Birth': 'datetime64[ns]', 
            'Source of Traffic': 'category', 
            'Onboard Wifi Service': 'Int64',
            'Embarkation/Disembarkation time convenient': 'Int64', 
            'Ease of Online booking': 'Int64',
            'Gate location': 'Int64', 
            'Logging': 'datetime64[ns]', 
            'Onboard Dining Service': 'Int64', 
            'Online Check-in': 'Int64',
            'Cabin Comfort': 'Int64', 
            'Onboard Entertainment': 'Int64', 
            'Cabin service': 'Int64',
            'Baggage handling': 'Int64', 
            'Port Check-in Service': 'Int64', 
            'Onboard Service': 'Int64',
            'Cleanliness': 'Int64', 
            'Cruise Name': 'category', 
            'Ticket Type': 'category', 
            'Cruise Distance': 'float64', 
            'WiFi': 'Int64',
            'Dining': 'Int64', 
            'Entertainment': 'Int64',
        })

        return df