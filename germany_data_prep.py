__author__ = 'tingcooper'

import datetime as dt
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt


pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 10000)


def get_data():
    #get and filter generation data for germany 2016
    df_gen = pd.read_csv('time_series_60min_singleindex.csv',
                         usecols=(lambda s: s.startswith('utc') |
                                 s.startswith('DE')),
                        parse_dates=[0], index_col=0)

    df_gen16 = df_gen.loc[df_gen.index.year == 2016, :]
    cols = ['DE_solar_generation_actual', 'DE_wind_generation_actual', 'DE_wind_offshore_generation_actual', 'DE_wind_onshore_generation_actual']
    df_gen16 = df_gen16[cols]

    #get weather data
    df_weather = pd.read_csv('weather_data_GER_2016.csv', index_col=0)
    df_daily_weather = df_weather.groupby(df_weather.index).mean()

    # merge generation and weather data
    df_gen_weather = pd.merge(df_gen16, df_daily_weather, how='left', left_index=True, right_index=True)
    df_gen_weather['dt'] = df_gen_weather.index

    w_stats = list(df_weather.columns.values[3:])
    for stat in w_stats:
        #stat = 'v1'
        print stat
        df_gen_weather[stat + '-6h'] = df_gen_weather['dt'].apply(lambda x: df_gen_weather.ix[df_gen_weather['dt'] == x - timedelta(hours=6)][stat].values)
        #df_gen_weather[stat + '-1d'] = df_gen_weather['dt'].apply(lambda x: get_past_weather(x, df_gen_weather, stat))
        df_temp = pd.DataFrame(df_gen_weather[stat + '-6h'].values.tolist())
        df_gen_weather[stat + '-6h'] = df_temp[0].values

    return df_gen_weather, df_weather

def make_features(df_gen_weather):
    df = df_gen_weather
    df['day_of_year'] = df['dt'].dt.dayofyear
    df['month'] = df['dt'].dt.month
    df['day_of_week'] = df['dt'].dt.dayofweek
    df['hr_of_day'] = df['dt'].dt.hour
    df['day_of_month'] = df['dt'].dt.dayofmonth

    return df

def do_all():
    df_gen_weather, df_weather = get_data()
    df_features = make_features(df_gen_weather)
    df_features.to_csv('germany_2016_with_features.csv',index=False)



def make_charts(df_all):
    df_all.plot(kind='bar', x='day_of_week', y='DE_wind_generation_actual')

'''
def make_map(df_weather):

    df = df_weather
    '''
    BBox = ((df.lon.min(),   df.lon.max(),
         df.lat.min(), df.lat.max()))
    df = df.reset_index()
    '''

    df = df[['lat', 'lon']]
    df = df.drop_duplicates()
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    gdf = GeoDataFrame(df, geometry=geometry)

    de = gpd.read_file('Germany_shapefile/de_10km.shp')
    fig,ax = plt.subplots(figsize=(15,15))
    de.plot(ax=ax)

    #gdf.plot(ax=de.plot(), figsize=(15,15))




def get_past_weather(x, df_gen_weather,stat):
    prev_day = x - timedelta(days=1)
    stat_val = df_gen_weather.ix[df_gen_weather['dt'] == prev_day][stat]
    stat_val = stat_val.values
    print stat_val
    #stats = row[row.columns[-11:]]
    return stat_val
'''


def plot_stuff():


    df_all = pd.read_csv('germany_2016_with_features.csv')

    ax = df_all.groupby('hr_of_day').DE_wind_generation_actual.mean().plot(kind='bar')
    ax.set_xlabel("hour of day")
    ax.set_ylabel("Average wind generation (MW)")

    ax = df_all.groupby('hr_of_day').v2.mean().plot(kind='bar')
    ax.set_xlabel("hour of day")
    ax.set_ylabel("Average wind speed at 2m above displacement height")
    ax.set_ylabel("Average wind speed at 10m above displacement height")
    ax.set_ylabel("Average wind speed at 50m above ground level")


    ax = df_all.groupby('month').DE_wind_generation_actual.mean().plot(kind='bar')
    ax.set_xlabel("month")
    ax.set_ylabel("Average wind generation (MWh)")

    ax = df_all.groupby('month').v_50m.mean().plot(kind='bar')
    ax.set_xlabel("month")
    ax.set_ylabel("Average wind speed at 2m above displacement height")
    ax.set_ylabel("Average wind speed at 10m above displacement height")
    ax.set_ylabel("Average wind speed at 50m above ground level")

    ax = df_all.groupby('day_of_year').DE_wind_generation_actual.mean().plot(kind='line')
    ax.set_xlabel("day of year")
    ax.set_ylabel("Average wind generation (MWh)")

    ax = df_all.groupby('day_of_year').v1.mean().plot(kind='line')
    ax.set_xlabel("day of year")
    ax.set_ylabel("Average wind speed at 2m above displacement height")
    ax.set_ylabel("Average wind speed at 10m above displacement height")
    ax.set_ylabel("Average wind speed at 50m above ground level")

    ax = df_all.groupby('day_of_month').DE_wind_generation_actual.mean().plot(kind='bar')
    ax.set_xlabel("day of month")
    ax.set_ylabel("Average wind generation (MWh)")

    ax = df_all.groupby('day_of_month').v_50m.mean().plot(kind='bar')
    ax.set_xlabel("day of month")
    ax.set_ylabel("Average wind speed at 2m above displacement height")
    ax.set_ylabel("Average wind speed at 10m above displacement height")
    ax.set_ylabel("Average wind speed at 50m above ground level")

    ax1= df_all.plot.scatter(y='DE_wind_generation_actual', x='v1',  style='o', c='green', legend=True)
    ax2 = df_all.plot.scatter(y='DE_wind_generation_actual', x='v2',  style='o', c='blue', ax=ax1, legend=True)
    ax3 = df_all.plot.scatter(y='DE_wind_generation_actual', x='v_50m',  style='o', c='purple', ax=ax1, legend=True)
    ax1.set_xlabel("average hourly wind speed (m/s)")
    ax1.set_ylabel("Average wind generation (MWh)")
    ax1.legend()
    ax1.show()
    plt.legend(loc='best')
    plt.show()



    print(ax1 == ax2 == ax3)  # True




