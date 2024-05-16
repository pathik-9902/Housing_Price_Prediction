import pandas as pd
import numpy as np

areas_data = {
    "Area1": {"ranking": 1, "locations": ["Adajan", "Pal", "Vesu"]},
    "Area2": {"ranking": 2, "locations": ["Athwa", "Ghod Dod Road", "City Light"]},
    "Area3": {"ranking": 3, "locations": ["Piplod", "Varachha", "Althan"]},
    "Area4": {"ranking": 4, "locations": ["Sarthana", "Katargam", "Udhna"]},
    "Area5": {"ranking": 5, "locations": ["Sachin", "Dindoli", "Bhestan"]}
}

areas = {area: data["ranking"] for area, data in areas_data.items()}
area_locations = {area: data["locations"] for area, data in areas_data.items()}

price_ranges = {
    "Area1": {year: (50 + 30 * (year - 2000)**1.01, 1000 + 40 * (year - 2000)**1.01) for year in range(2000, 2024)},
    "Area2": {year: (40 + 20 * (year - 2000)**1.01, 900 + 30 * (year - 2000)**1.01) for year in range(2000, 2024)},
    "Area3": {year: (30 + 20 * (year - 2000)**1.01, 800 + 30 * (year - 2000)**1.01) for year in range(2000, 2024)},
    "Area4": {year: (25 + 20 * (year - 2000)**1.01, 700 + 30 * (year - 2000)**1.01) for year in range(2000, 2024)},
    "Area5": {year: (10 + 10 * (year - 2000)**1.01, 500 + 20 * (year - 2000)**1.01) for year in range(2000, 2024)}
}

crime_rate_trends = {
    "Area1": np.linspace(8, 2, num=24), 
    "Area2": np.linspace(10, 4, num=24),
    "Area3": np.linspace(12, 6, num=24),
    "Area4": np.linspace(14, 8, num=24),
    "Area5": np.linspace(16, 10, num=24)
}

def classify_area(location):
    for area, locations in area_locations.items():
        if location in locations:
            return area
    return "Other"

data = []
for year in range(2000, 2024):
    for _ in range(1000):
        area = np.random.choice(list(areas.keys()))
        location = np.random.choice(area_locations[area])
        sqft = np.random.randint(1000, 3001)
        crime_rates = crime_rate_trends[area][year - 2000]
        amenities = np.random.randint(1, 11)
        real_estate_type = np.random.choice(['Apartment', 'House', 'Villa'])
        data.append([year, location, sqft, amenities, real_estate_type, crime_rates, area])

df = pd.DataFrame(data, columns=['year', 'location', 'sqft', 'amenities', 'real_estate_type', 'crime_rates', 'area'])

data = {
    'year': [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001, 2000],
    'Population': [8331000, 8065000, 7784000, 7490000, 7185000, 6874000, 6564000, 6251000, 5954000, 5671000, 5401000, 5144000, 4900000, 4667000, 4445000, 4233000, 4032000, 3840000, 3658000, 3484000, 3318000, 3160000, 3010000, 2867000, 2867000],
    'GDP (in crores)': [25630, 22620, 19440, 16590, 16500, 15030, 13290, 11670, 10290, 8080, 7240, 6160, 455, 365, 330, 228, 213, 189, 6, 153, 149, 122, 113, 107, 107],
    'Fixed Deposit Interest Rate (% p.a.)': [5.35, 5.90, 5.35, 5.35, 5.35, 5.70, 6.25, 6.25, 6.50, 7.00, 8.50, 8.75, 8.75, 9.00, 8.75, 6.50, 7.75, 7.50, 7.75, 6.25, 5.75, 5.25, 5.50, 8.50, 9.50],
    'Literacy Rate (%)': [87.89, 87.30, 86.70, 86.15, 85.65, 85.20, 84.80, 84.45, 84.15, 83.90, 83.70, 83.55, 83.45, 83.40, 83.40, 83.40, 83.30, 83.00, 82.50, 81.80, 80.90, 79.80, 78.50, 77.00, 75.00]
}

new_data = pd.DataFrame(data)

df = df.merge(new_data, on='year', how='left')

df['Literacy Rate (%)'] = df['Literacy Rate (%)'].interpolate()

df['GDP (in crores)'] = df['GDP (in crores)'].interpolate()

df['Fixed Deposit Interest Rate (% p.a.)'] = df['Fixed Deposit Interest Rate (% p.a.)'].interpolate()

df['Population'] = df['Population'].interpolate()

df['Development Rate'] = np.exp(0.2 * (df['year'] - 2000)) + 0.1

df = df.round(2)

df['HF'] = ((df['Population']/5000000) * (df['GDP (in crores)']/1000) * df['Literacy Rate (%)'] * df['Development Rate']) / (df['crime_rates'] * df['Fixed Deposit Interest Rate (% p.a.)'])

df['HF'] = df['HF'].round(2)

type_priority = {
    "Apartment": 10,
    "House": 20,
    "Villa": 30
}

df['Area-wise Price'] = df.apply(lambda row: price_ranges[row['area']][row['year']][0], axis=1)

df['price'] = ((df['Area-wise Price'] * df['sqft']) * ((df['amenities']) + df['real_estate_type'].map(type_priority)))*df['HF']**0.01

df['price'] = df['price'].round(2)

df.to_csv('Backend/housing_data.csv', index=False)

print(df.head())
print(df.tail())

print(df.shape)
