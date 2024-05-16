import pandas as pd

df = pd.read_csv("Backend/housing_data.csv")

areas_data = {
    "Area1": {"ranking": 1, "locations": ["Adajan", "Pal", "Vesu"]},
    "Area2": {"ranking": 2, "locations": ["Athwa", "Ghod Dod Road", "City Light"]},
    "Area3": {"ranking": 3, "locations": ["Piplod", "Varachha", "Althan"]},
    "Area4": {"ranking": 4, "locations": ["Sarthana", "Katargam", "Udhna"]},
    "Area5": {"ranking": 5, "locations": ["Sachin", "Dindoli", "Bhestan"]}
}

area_mapping = {location: area for area, data in areas_data.items() for location in data["locations"]}
df['location'] = df['location'].map(lambda x: int(area_mapping[x][-1]))

type_priority = {
    "Apartment": 1,
    "House": 2,
    "Villa": 3
}

df['real_estate_type'] = df['real_estate_type'].map(type_priority)

df['HF'] = ((df['Population']/500000) * (df['GDP (in crores)']/100) * df['Literacy Rate (%)'] * df['Development Rate']) / (df['crime_rates'] * df['Fixed Deposit Interest Rate (% p.a.)'])

df.drop(['Population','Area-wise Price', 'area', 'GDP (in crores)', 'Fixed Deposit Interest Rate (% p.a.)', 'Literacy Rate (%)', 'Development Rate', 'crime_rates'], axis=1, inplace=True)

df = df.round(2)

print(df.dtypes)

df.to_csv("Backend/processed.csv", index=False)
