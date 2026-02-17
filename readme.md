# Car Price Prediction Project

## What is this project about

I am working on a data analysis project where i am trying to understand how car prices work and what things affect the price of a car. I have a dataset of 2500 cars with information like brand, year, mileage, engine size, fuel type, transmission, condition and price. My goal is to explore this data first and then later build a model that can predict car prices.

---

## Dataset

The dataset i am using is called `caar.csv`. It has 2500 rows and 10 columns. Each row is one car.

The columns in the dataset are:

- **Car ID** - unique number for each car
- **Brand** - the company that made the car like Toyota, BMW, Audi etc
- **Year** - the year the car was made
- **Engine Size** - size of the car engine
- **Fuel Type** - petrol, diesel, electric or hybrid
- **Transmission** - manual or automatic
- **Mileage** - how many miles the car has been driven
- **Condition** - is the car new, used or like new
- **Price** - the price of the car in dollars
- **Model** - the specific model name of the car

---

## What i have done so far

### Step 1 - Sorted the data by year

The first thing i noticed is that the data was not in any order. The rows were random, like year 2016 then 2001 then 2019. So i sorted all 2500 rows by year from 2000 to 2023 so it is in proper order. I also re-numbered the Car ID column from 1 to 2500 to match the new order. I saved this as a new file called `caar_sorted.csv`.

```python
import pandas as pd

df = pd.read_csv('caar.csv')
df = df.sort_values('Year').reset_index(drop=True)
df['Car ID'] = range(1, len(df) + 1)
df.to_csv('caar_sorted.csv', index=False)
```

### Step 2 - Checked how car price changes over the years

I wanted to see if car prices are going up or down over the years. So i grouped all the cars by year and calculated the average price for each year. Then i plotted it on a graph with year on x axis and price on y axis.

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('caar_sorted.csv')
avg_price = df.groupby('Year')['Price'].mean()

plt.plot(avg_price.index, avg_price.values)
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Car Price by Year')
plt.show()
```

### Step 3 - Checked price trend for each brand separately

There are 7 car brands in the dataset which are Toyota, BMW, Audi, Mercedes, Tesla, Honda and Ford. I wanted to see how the price changes over years for each brand separately. So i made a separate graph for each brand.

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('caar_sorted.csv')
brands = df['Brand'].unique()

for brand in brands:
    brand_df = df[df['Brand'] == brand]
    avg_price = brand_df.groupby('Year')['Price'].mean()

    plt.figure()
    plt.plot(avg_price.index, avg_price.values)
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.title(f'{brand} Car Price by Year')
    plt.show()
```

### Step 4 - Checked mileage trend for each brand separately

Same thing i did for mileage. I wanted to see how the average mileage of cars changes over the years for each brand. This helps understand if newer cars have more or less mileage.

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('caar_sorted.csv')
brands = df['Brand'].unique()

for brand in brands:
    brand_df = df[df['Brand'] == brand]
    avg_mileage = brand_df.groupby('Year')['Mileage'].mean()

    plt.figure()
    plt.plot(avg_mileage.index, avg_mileage.values)
    plt.xlabel('Year')
    plt.ylabel('Mileage')
    plt.title(f'{brand} Mileage by Year')
    plt.show()
```

---

## Folder Structure

```
caar_price_analysis/
│
├── data/
│   ├── caar.csv
│   
│
├── notebooks/
│   └── jupyter note book.ipynb
│
└── README.md
```

---

## Libraries i am using

- `pandas` - for loading and working with the data
- `matplotlib` - for making graphs and charts

---

## What i plan to do next

- Explore more columns like fuel type, condition and transmission
- Find which things affect the price the most
- Build a machine learning model to predict car prices