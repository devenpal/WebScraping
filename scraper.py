# Uncomment following lines to install on terminal if you don't have these packages installed
# Or use requirements.txt
#!pip install bs4
#!pip install pandas
#!pip install sodapy
#!pip install gmplot

#Packages
from bs4 import BeautifulSoup
import requests
import sqlite3
import csv
import pandas as pd
import Lab6Code
from Lab6Code import csv_dataset
from sodapy import Socrata
import argparse
import math
from math import sin, cos, sqrt, atan2, radians
import gmplot
import numpy as np
from statistics import mean
from scipy import stats
import matplotlib.pyplot as plt

#Selenium packages
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

content = requests.get('https://homicide.latimes.com/')
soup = BeautifulSoup(content.content, 'html.parser')

# Gets HTML website information using Beautiful Soup
def get_LATimes_website():
    content = requests.get('https://homicide.latimes.com/')
    soup = BeautifulSoup(content.content, 'html.parser')
    results = soup.find(id='container')

    #print(results)
    #print(soup.prettify())

# Gets victim information for name and age
def get_LATimes_victiminfo():
    #content = requests.get('https://homicide.latimes.com/')
    #soup = BeautifulSoup(content.content, 'html.parser')

    results = soup.find(id='container')
    h_tags = soup_selenium.findAll('h2')
    #print(h_tags)

    htag_list = []
    for i in range(2, len(h_tags)):
        htag_list.append(h_tags[i].text.strip())

    #print(htag_list)
    #print(len(htag_list))
    #print()

    age_list = []
    name_list = []

    for i in range(len(htag_list)):
        split_list = htag_list[i].split(', ')
        name_list.append(split_list[0])
        age_list.append(split_list[1])

    #print()
    #print(name_list)
    #print(len(name_list))

    rows = zip(name_list, age_list)

    with open('LATimes.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(("Victim Name", "Vict Age"))
        for row in rows:
            writer.writerow(row)

# Gets homicide date info for each victim
def get_LATimes_homicidedates():
    #content = requests.get('https://homicide.latimes.com/')
    #soup = BeautifulSoup(content.content, 'html.parser')

    results = soup_selenium.findAll('div', attrs={"class": "death-date"})
    #print((results))

    death_date_list = []

    for i in range(len(results)):
        death_date_list.append(results[i].text.strip())

    #print(death_date_list)

    df = pd.read_csv("LATimes.csv")
    df["DATE OCC"] = death_date_list
    df.to_csv("LATimes.csv", index=False)

# Gets address and neighborhood of each homicide victim
def get_LATimes_homicidelocations():
    #results = soup_selenium.findAll('div', attrs={"class": "post-list-badge"})
    #print(results)

    neighbor_list=[]
    addy_list=[]
    for divtag in soup_selenium.findAll('div', attrs={"class": "post-list-badge"}):
        for ultag in divtag.find_all('ul', {'class': 'badge-location'}):
            if ultag.a == None:
                continue
                #print((ultag))
            else:
                neighbor_list.append(ultag.a.text.strip())
        if(divtag):
            link = divtag.a
            if(link==None):
                neighbor_list.append(None)


    for divtag in soup_selenium.findAll('div', attrs={"class": "post-list-badge"}):
        for ultag in divtag.find_all('ul', {'class': 'badge-location'}):
            for litag in ultag.find_all('li'):
                if litag.text.strip() not in neighbor_list:
                    addy_list.append(litag.text.strip())
                if not litag.text:
                    continue

        if (divtag):
            link = divtag.a
            if (link == None):
                for litag in ultag.find_all('li'):
                    if litag.text.strip() not in addy_list or litag.text==None:
                        addy_list.append(None)

    #print(addy_list)
    #print(len(addy_list))

    #print(neighbor_list)
    #print(len(neighbor_list))

    df = pd.read_csv("LATimes.csv")
    df["LOCATION"] = addy_list  # address
    df["AREA NAME"] = neighbor_list  # neighborhood
    df.to_csv("LATimes.csv", index=False)

# Gets location of each homicide for LATimes victims using Google Maps API to reverse geocode
def get_LATimes_latlon_info():
    apikey= 'AIzaSyCU9QR8FL_oxNsSoAUH893d47nq4jKOvIk'
    df = pd.read_csv("LATimes.csv")
    df = df.astype(object).replace(np.nan, 'None')
    csv_location_list = df['LOCATION'].tolist()
    csv_area_list = df['AREA NAME'].tolist()
    #print(csv_area_list)
    #print(csv_location_list)
    list_of_lists = [(csv_location_list[i], csv_area_list[i]) for i in range(0, len(csv_location_list))]
    #print(list_of_lists)
    street_list = []
    lat_list = []
    lon_list = []

    for i in range(len(list_of_lists)):
        if list_of_lists[i][0] == 'None' or list_of_lists[i][1] == 'None':
            lat_list.append(None)
            lon_list.append(None)

        else:
            location_results = gmplot.GoogleMapPlotter.geocode(list_of_lists[i][0] + ', ' + list_of_lists[i][1], apikey=apikey)
            #print(location_results)

            if location_results and len(location_results):
                #street_list.append(str(csv_location_list[i][0]))
                lat_list.append(location_results[0])
                lon_list.append(location_results[1])
                #print(location_results)

            else:
                lat_list.append(None)
                lon_list.append(None)
                print("ALSO EXECUTE")

    #print(len(lat_list))
    #print(len(lon_list))
    #print(street_list)
    #print(len(street_list))

    list_of_tuples = [(lat_list[i], lon_list[i]) for i in range(0, len(lat_list))]
    #print(list_of_tuples)

    df = pd.read_csv("LATimes.csv")
    df["lat"] = lat_list
    df["lon"] = lon_list
    df.to_csv("LATimes.csv", index=False)

    rows = zip(lat_list, lon_list)
    with open('LATimes_latlon.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(("latitude", "longitude"))
        for row in rows:
            writer.writerow(row)

def LATimes_to_database():
    csv_data = csv_dataset('LATimes.csv', ',')
    con = sqlite3.connect('LAhomicides.db')
    cur = con.cursor()

    cur.execute('DROP TABLE IF EXISTS LATimes')

    cur.execute('''CREATE TABLE LATimes
                    (victim_name text, vict_age integer, date_occ text, location text, area_name text, lat real, lon real)''')

    for row in csv_data[1:]:
        cur.execute('INSERT INTO LATimes VALUES (?, ?, ?, ?, ?, ?, ?)', row)

    con.commit()
    con.close()

def LATimes_latlon_to_database():
    csv_data = csv_dataset('LATimes_latlon.csv', ',')
    con = sqlite3.connect('LAhomicides.db')
    cur = con.cursor()

    cur.execute('DROP TABLE IF EXISTS LATimes_latlon')

    cur.execute('''CREATE TABLE LATimes_latlon
                        (lat real, lon real)''')

    for row in csv_data[1:]:
        cur.execute('INSERT INTO LATimes_latlon VALUES (?, ?)', row)

    con.commit()
    con.close()

def get_LAPublicCrime(entries):
    client = Socrata("data.lacity.org")

    results = client.get("2nrs-mtv8", limit=entries)
    # print(results)
    results_df = pd.DataFrame.from_records(results)
    # print(results_df['date_occ'])
    #print(results_df.head())

    results_df.to_csv('LAPublicCrimeData.csv', index=False)

def LAPublicCrime_to_database():
    csv_data = csv_dataset('LAPublicCrimeData.csv', ',')
    con = sqlite3.connect('LAhomicides.db')
    cur = con.cursor()

    cur.execute('DROP TABLE IF EXISTS LAPublicCrimeData')

    df = pd.read_csv('LAPublicCrimeData.csv')
    df.to_sql('LAPublicCrimeData', con, if_exists='append', index=False)

    con.commit()
    con.close()

def dist(point1, point2):
    R = 6373.0
    lat1 = radians(point1[1])
    lon1 = radians(point1[0])
    lat2 = radians(point2[1])
    lon2 = radians(point2[0])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def LAPublicCrime_to_dict_static():
    pathname = os.path.join(args.static, 'All_LAhomicides.db')
    con = sqlite3.connect(pathname)
    cur = con.cursor()

    cur.execute('SELECT dr_no FROM LAPublicCrimeData')
    vict_no = cur.fetchall()
    cur.execute('SELECT lat FROM LAPublicCrimeData')
    lat = cur.fetchall()
    cur.execute('SELECT lon FROM LAPublicCrimeData')
    lon = cur.fetchall()

    for i in range(len(lat)):
        lat[i] = lat[i][0]

    for i in range(len(lon)):
        lon[i] = lon[i][0]

    for i in range(len(vict_no)):
        vict_no[i] = vict_no[i][0]

    # print(vict_no)
    lon_lat_list = list(zip(lon, lat))
    vic_dict = dict(zip(vict_no, lon_lat_list))

    return vic_dict

def LAPublicCrime_to_dict_scraper():
    con = sqlite3.connect('LAhomicides.db')
    cur = con.cursor()

    cur.execute('SELECT dr_no FROM LAPublicCrimeData')
    vict_no = cur.fetchall()
    cur.execute('SELECT lat FROM LAPublicCrimeData')
    lat = cur.fetchall()
    cur.execute('SELECT lon FROM LAPublicCrimeData')
    lon = cur.fetchall()

    for i in range(len(lat)):
        lat[i] = lat[i][0]

    for i in range(len(lon)):
        lon[i] = lon[i][0]

    for i in range(len(vict_no)):
        vict_no[i] = vict_no[i][0]

    # print(vict_no)
    lon_lat_list = list(zip(lon, lat))
    vic_dict = dict(zip(vict_no, lon_lat_list))

    return vic_dict

def distance_analysis_static():
    apikey = 'AIzaSyCU9QR8FL_oxNsSoAUH893d47nq4jKOvIk'
    pathname = os.path.join(args.static, 'All_LAhomicides.db')
    con = sqlite3.connect(pathname)
    cur = con.cursor()
    vict_info=LAPublicCrime_to_dict_static()

    # print(list(vic_dict.items())[0])
    cur.execute('SELECT victim_name FROM LATimes')
    vict_name = cur.fetchall()
    for i in range(len(vict_name)):
        vict_name[i] = vict_name[i][0]

    cur.execute('SELECT lat FROM LATimes')
    lat_LATimes = cur.fetchall()
    for i in range(len(lat_LATimes)):
        lat_LATimes[i] = lat_LATimes[i][0]

    cur.execute('SELECT lon FROM LATimes')
    lon_LATimes = cur.fetchall()
    for i in range(len(lon_LATimes)):
        lon_LATimes[i] = lon_LATimes[i][0]

    mn = float('inf')
    nearest = ''

    nearest_list = []
    final_list = []
    dist_list = []

    for i in range(len(lon_LATimes)):
        for k, v in vict_info.items():
            if (lon_LATimes[i] == '' or lat_LATimes[i] == ''):
                continue

            if mn > dist(v, (lon_LATimes[i], lat_LATimes[i])):
                mn = dist(v, (lon_LATimes[i], lat_LATimes[i]))
                nearest = k

        # if mn<0.4 you can assume it's the same crime
        if (mn < 0.4):
            nearest_list.append(None)
        else:
            nearest_list.append(nearest)
        dist_list.append(mn)
        # print(nearest, mn)
        mn = float('inf')

    # print(len(nearest_list))
    print()
    print()
    print('ANALYSIS SECTION: ')
    print()
    print('The number of LA Times homicides that match with the LAPublicCrime database are: ' + str(
        len([x for x in nearest_list if x is None])) + " out of " + str(len(nearest_list)))

    crime_list = []
    for i in range(len(nearest_list)):
        if ((nearest_list[i]) == None):
            continue
        else:
            # print(str(nearest_list[i]))
            cur.execute("SELECT crm_cd_desc from LAPublicCrimeData WHERE dr_no is (?)", [str(nearest_list[i])])
        results = cur.fetchall()
        crime_list.append(results[0][0])

    assault_list = []
    homicide_list = []
    shooting_list = []
    robbery_list = []
    weapon_list = []

    for i in range(len(crime_list)):
        if 'HOMICIDE' in crime_list[i]:
            homicide_list.append(crime_list[i])

        if 'SHOTS' in crime_list[i]:
            shooting_list.append(crime_list[i])

        if 'BRANDISH WEAPON' in crime_list[i]:
            weapon_list.append(crime_list[i])

        if 'ASSAULT' in crime_list[i]:
            assault_list.append(crime_list[i])

        if 'ROBBERY' in crime_list[i]:
            robbery_list.append(crime_list[i])

    print()
    print('Number of homicides near other homicides: ' + str(len(homicide_list)))
    print()
    print('Number of shootings near other homicides: ' + str(len(shooting_list)))
    print()
    print('Number of assaults near other homicides: ' + str(len(assault_list)))
    print()
    print('Number of robberies near other homicides: ' + str(len(robbery_list)))
    print()
    print('Number of crimes involving threat of weapons near other homicides: ' + str(len(weapon_list)))
    print()

    total_violent_crimes = len(homicide_list) + len(shooting_list) + len(assault_list) + len(robbery_list) + len(
        weapon_list)
    total_nonmatched = len(nearest_list) - len([x for x in nearest_list if x is None])
    percentage = (total_violent_crimes / total_nonmatched) * 100

    print('Number of violent crimes near other ' + str(
        len(nearest_list) - len([x for x in nearest_list if x is None])) + ' homicides from LA Times that are not matched with LAPublicCrime database: ' + str(
        len(homicide_list) + len(shooting_list) + len(assault_list) + len(robbery_list) + len(weapon_list)))
    print()
    print(str(round(percentage, 2)) + '% of nearest crimes to homicides are other violent crimes')
    print()

    # Visualization
    crime_names = ['Homicides', 'Shootings', 'Assaults', 'Robberies', 'Weapons Crimes']
    no_of_crimes = [len(homicide_list), len(shooting_list), len(assault_list), len(robbery_list), len(weapon_list)]
    plt.bar(crime_names, no_of_crimes)
    plt.ylabel('# of crimes')
    plt.xlabel('Type of crime')
    plt.title('Type of crimes nearest to homicides')
    print('First visualization based on analysis titled: Type of crimes nearest to homicides')
    print()
    plt.show()

    mean_list = []
    for i in range(len(dist_list)):
        if (dist_list[i] < float('inf')):
            mean_list.append(dist_list[i])

    print('Mean distance is: ' + str(mean(mean_list)))
    numpy_array = np.array(mean_list)
    print('Trimmed mean distance, accounting for 10% outliers is : ' + str(stats.trim_mean(numpy_array, 0.1)))
    print()

    cur.execute('SELECT victim_name FROM LATimes')
    victim_name = cur.fetchall()
    cur.execute('SELECT vict_age FROM LATimes')
    vict_age = cur.fetchall()
    cur.execute('SELECT lat FROM LATimes')
    lat = cur.fetchall()
    cur.execute('SELECT lon FROM LATimes')
    lon = cur.fetchall()

    gmap = gmplot.GoogleMapPlotter(34, -118, 10, apikey=apikey)

    # Plot first 5 homicides from LATimes db
    for i in range(5):
        gmap.marker(lat[i][0], lon[i][0], color='green', info_window=victim_name[i][0] + ", " + str(vict_age[i][0]))

    for i in range(5):  # len(nearest_list)
        if (vict_info.get(nearest_list[i]) == None):
            gmap.marker(lat[i][0], lon[i][0], label='Match',
                        info_window=str(nearest) + ": Match in public crime database to " + vict_name[i])
        else:
            gmap.marker(vict_info.get(nearest_list[i])[1], vict_info.get(nearest_list[i])[0],
                        info_window=str(nearest_list[i]) + ": Nearest crime to " + vict_name[i])
            # print(vict_name[i], nearest_list[i])

    print("Displaying distance for 2 nearest crimes of non-matched crimes for: " + str(
        len([x for x in nearest_list if x is not None]) - len([x for x in dist_list if x == float('inf')])) + " crimes")

    for i in range(len(dist_list)):
        if (dist_list[i] >= 0.4 and dist_list[i] < float('inf')):
            print(
                'Distance between 2 nearest crimes of ' + str(nearest_list[i]) + ' and ' + vict_name[i] + " is " + str(
                    dist_list[i]))

    # print(mn)
    # print(nearest)
    print()
    print("Check folder for map titled map1.html of first 5 crimes in LATimes")
    gmap.draw('map1.html')

    con.commit()
    con.close()

def distance_analysis_scraper():
    apikey = 'AIzaSyCU9QR8FL_oxNsSoAUH893d47nq4jKOvIk'
    con = sqlite3.connect('LAhomicides.db')
    cur = con.cursor()
    vict_info=LAPublicCrime_to_dict_scraper()

    # print(list(vic_dict.items())[0])
    cur.execute('SELECT victim_name FROM LATimes')
    vict_name = cur.fetchall()
    for i in range(len(vict_name)):
        vict_name[i] = vict_name[i][0]

    cur.execute('SELECT lat FROM LATimes')
    lat_LATimes = cur.fetchall()
    for i in range(len(lat_LATimes)):
        lat_LATimes[i] = lat_LATimes[i][0]

    cur.execute('SELECT lon FROM LATimes')
    lon_LATimes = cur.fetchall()
    for i in range(len(lon_LATimes)):
        lon_LATimes[i] = lon_LATimes[i][0]

    mn = float('inf')
    nearest = ''

    nearest_list = []
    final_list = []
    dist_list = []

    for i in range(len(lon_LATimes)):
        for k, v in vict_info.items():
            if (lon_LATimes[i] == '' or lat_LATimes[i] == ''):
                continue

            if mn > dist(v, (lon_LATimes[i], lat_LATimes[i])):
                mn = dist(v, (lon_LATimes[i], lat_LATimes[i]))
                nearest = k

        # if mn<0.4 you can assume it's the same crime
        if (mn < 0.4):
            nearest_list.append(None)
        else:
            nearest_list.append(nearest)
        dist_list.append(mn)
        # print(nearest, mn)
        mn = float('inf')

    # print(len(nearest_list))
    print()
    print()
    print('ANALYSIS SECTION: ')
    print()
    print('The number of LA Times homicides that match with the LAPublicCrime database are: ' + str(
        len([x for x in nearest_list if x is None])) + " out of " + str(len(nearest_list)))

    crime_list = []
    for i in range(len(nearest_list)):
        if ((nearest_list[i]) == None):
            continue
        else:
            # print(str(nearest_list[i]))
            cur.execute("SELECT crm_cd_desc from LAPublicCrimeData WHERE dr_no is (?)", [str(nearest_list[i])])
        results = cur.fetchall()
        crime_list.append(results[0][0])

    assault_list = []
    homicide_list = []
    shooting_list = []
    robbery_list = []
    weapon_list = []

    for i in range(len(crime_list)):
        if 'HOMICIDE' in crime_list[i]:
            homicide_list.append(crime_list[i])

        if 'SHOTS' in crime_list[i]:
            shooting_list.append(crime_list[i])

        if 'BRANDISH WEAPON' in crime_list[i]:
            weapon_list.append(crime_list[i])

        if 'ASSAULT' in crime_list[i]:
            assault_list.append(crime_list[i])

        if 'ROBBERY' in crime_list[i]:
            robbery_list.append(crime_list[i])

    print()
    print('Number of homicides near other homicides: ' + str(len(homicide_list)))
    print()
    print('Number of shootings near other homicides: ' + str(len(shooting_list)))
    print()
    print('Number of assaults near other homicides: ' + str(len(assault_list)))
    print()
    print('Number of robberies near other homicides: ' + str(len(robbery_list)))
    print()
    print('Number of crimes involving threat of weapons near other homicides: ' + str(len(weapon_list)))
    print()

    total_violent_crimes = len(homicide_list) + len(shooting_list) + len(assault_list) + len(robbery_list) + len(
        weapon_list)
    total_nonmatched = len(nearest_list) - len([x for x in nearest_list if x is None])
    percentage = (total_violent_crimes / total_nonmatched) * 100

    print('Number of violent crimes near other ' + str(
        len(nearest_list) - len([x for x in nearest_list if x is None])) + ' homicides from LA Times that are not matched with LAPublicCrime database: ' + str(
        len(homicide_list) + len(shooting_list) + len(assault_list) + len(robbery_list) + len(weapon_list)))
    print()
    print(str(round(percentage, 2)) + '% of nearest crimes to homicides are other violent crimes')
    print()

    # Visualization
    crime_names = ['Homicides', 'Shootings', 'Assaults', 'Robberies', 'Weapons Crimes']
    no_of_crimes = [len(homicide_list), len(shooting_list), len(assault_list), len(robbery_list), len(weapon_list)]
    plt.bar(crime_names, no_of_crimes)
    plt.ylabel('# of crimes')
    plt.xlabel('Type of crime')
    plt.title('Type of crimes nearest to homicides')
    print('First visualization based on analysis titled: Type of crimes nearest to homicides')
    print()
    plt.show()

    mean_list = []
    for i in range(len(dist_list)):
        if (dist_list[i] < float('inf')):
            mean_list.append(dist_list[i])

    print('Mean distance is: ' + str(mean(mean_list)))
    numpy_array = np.array(mean_list)
    print('Trimmed mean distance, accounting for 10% outliers is : ' + str(stats.trim_mean(numpy_array, 0.1)))
    print()

    cur.execute('SELECT victim_name FROM LATimes')
    victim_name = cur.fetchall()
    cur.execute('SELECT vict_age FROM LATimes')
    vict_age = cur.fetchall()
    cur.execute('SELECT lat FROM LATimes')
    lat = cur.fetchall()
    cur.execute('SELECT lon FROM LATimes')
    lon = cur.fetchall()

    gmap = gmplot.GoogleMapPlotter(34, -118, 10, apikey=apikey)

    # Plot first 5 homicides from LATimes db
    for i in range(5):
        gmap.marker(lat[i][0], lon[i][0], color='green', info_window=victim_name[i][0] + ", " + str(vict_age[i][0]))

    for i in range(5):  # len(nearest_list)
        if (vict_info.get(nearest_list[i]) == None):
            gmap.marker(lat[i][0], lon[i][0], label='Match',
                        info_window=str(nearest) + ": Match in public crime database to " + vict_name[i])
        else:
            gmap.marker(vict_info.get(nearest_list[i])[1], vict_info.get(nearest_list[i])[0],
                        info_window=str(nearest_list[i]) + ": Nearest crime to " + vict_name[i])
            # print(vict_name[i], nearest_list[i])

    print("Displaying distance for 2 nearest crimes of non-matched crimes for: " + str(
        len([x for x in nearest_list if x is not None]) - len([x for x in dist_list if x == float('inf')])) + " crimes")

    for i in range(len(dist_list)):
        if (dist_list[i] >= 0.4 and dist_list[i] < float('inf')):
            print(
                'Distance between 2 nearest crimes of ' + str(nearest_list[i]) + ' and ' + vict_name[i] + " is " + str(
                    dist_list[i]))

    # print(mn)
    # print(nearest)
    print()
    print("Check folder for map titled map1.html of first 5 crimes in LATimes")
    gmap.draw('map1.html')

    con.commit()
    con.close()

parser=argparse.ArgumentParser()
parser.add_argument('--static', type=str)
args=parser.parse_args()

if __name__ == '__main__':
    if args.static:
        #print out stats from db file

        pathname = os.path.join(args.static, 'All_LAhomicides.db')
        con = sqlite3.connect(pathname)
        cur = con.cursor()
        print()
        print('May take around 3-5 minutes to run. map1.html and matplotlib visualization are outputted for analysis')
        print()
        print('First Dataset')
        print('Printing first 3 rows of the LA Times database table: ')
        cur.execute('SELECT * FROM LATimes LIMIT 3')
        results=cur.fetchall()
        print(results)

        print()
        print('Length of LA Times database table: ')
        cur.execute('SELECT * FROM LATimes')
        results=cur.fetchall()
        print(len(results))

        print()
        print('Second Dataset')
        print('Printing first 3 rows of the LA Public Crime Data database table: ')
        cur.execute('SELECT * FROM LAPublicCrimeData LIMIT 3')
        results=cur.fetchall()
        print(results)

        print()
        print('Length of LA Public Crime Data database table: ')
        cur.execute('SELECT * FROM LAPublicCrimeData')
        results=cur.fetchall()
        print(len(results))

        print()
        print('Third Dataset')
        print('Printing first 3 rows of the LA Times latitude and longitude database table: ')
        cur.execute('SELECT * FROM LATimes_latlon LIMIT 3')
        results = cur.fetchall()
        print(results)

        print()
        print('Length of LA Times latitude and longitude database table: ')
        cur.execute('SELECT * FROM LATimes_latlon')
        results=cur.fetchall()
        print(len(results))

        con.commit()
        con.close()

        distance_analysis_static()

    else:
        #Runs in the case of scraper.py

        driver = webdriver.Chrome()
        driver.get('https://homicide.latimes.com/')

        #Change scroll number if you want to edit the number of data points scraped
        ScrollNumber = 30
        time.sleep(2)
        for i in range(1, ScrollNumber):
            driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
            time.sleep(0.5)

        file = open('DS.html', 'w')
        file.write(driver.page_source)
        file.close()
        soup_selenium = BeautifulSoup(driver.page_source, "html.parser")

        driver.close()

        print()
        print('May take around 3-5 minutes to run. map1.html and matplotlib visualization are outputted for analysis')
        print()
        print('First Dataset')
        get_LATimes_victiminfo()
        get_LATimes_homicidedates()
        get_LATimes_homicidelocations()
        get_LATimes_latlon_info()
        LATimes = pd.read_csv('LATimes.csv')
        print('LATimes info: ')
        print(LATimes.info())
        print()
        print('LATimes shape: ')
        print(LATimes.shape)
        print()
        print('Printing first 10 entries of the LA Times homicide page: ')
        print(LATimes.head(10))
        LATimes_to_database()

        #Approximately 253,000 entries in the dataset. Enter around 250,000 entries for accurate results
        #Should take around 30 seconds to run

        get_LAPublicCrime(250000)
        LAPD = pd.read_csv('LAPublicCrimeData.csv')

        print()
        print('Second Dataset')
        print('LA Public Crime info:')
        print(LAPD.info())
        print()
        print('LA Public Crime shape:')
        print(LAPD.shape)
        print()
        print('Printing the first 10 entries of the LAPD Public Crime Dataset: ')
        print(LAPD.head(10))
        LAPublicCrime_to_database()

        print()
        print('Third Dataset')
        LATimes_lat_lon = pd.read_csv('LATimes_latlon.csv')
        print('LA Times latitude and longitude info:')
        print(LATimes_lat_lon.info())
        print()
        print('LA Times latitude and longitude shape:')
        print(LATimes_lat_lon.shape)
        print()
        print('Printing the first 10 entries of the LA Times latitude and longitude Dataset: ')
        print(LATimes_lat_lon.head(10))
        LATimes_latlon_to_database()

        distance_analysis_scraper()
