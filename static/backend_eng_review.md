# 📝 A Review Of Backend Engineering Jobs (as of Nov 18 2025)

This notebook uses [a pre-existing tool](https://github.com/speedyapply/JobSpy) to scrape the most recent backend engineering jobs from areas adjacent to major cities across the US, covering as much state as possible. The choice of cities is somewhat arbitrary. It has 3060 jobs from the past months, each job has the following attributes:

- title
- location
- area
- company
- date posted
- job type
- is remote
- job level
- job function
- description

The city and state name used to search for jobs is added as the `area` of jobs. It could be different from the exact location of the job. 

**Notes 0.1**: there were more jobs on the market, but the scrapper only scanned the some of the jobs. I didn't read in depth how the scrapper decide which job to keep.  
**Notes 0.2**: to verify the data aggregation pipeline and a list of cities covered in the dataset, check appendix at the end of the notebook.   
**Notes 0.3**: all data are available on Kaggle: [**view and download**](https://www.kaggle.com/datasets/tianyimasf/backend-engineer-jobs-us/data)

## 🔍 Exploratory Data Analysis

For this analysis, I only scraped jobs from LinkedIn. Indeed is weird to use and Google Job's data is usually dirty, and the jobs postings have a lot of overlaps anyways. 

From a first glance, we can see that there're a lot of null values in a lot of the fields. But if you look at the field names closely, most of the fields are not that important anyways. It's good to know them if exists, but not necessary to get a snapshot of what kind of backend engineering jobs are out there and what they require.


```python
import pandas as pd
import numpy as np
import os
```


```python
root_dir = "data/"
job_dir_backend_eng = "backend_engineer/"
job_dir_ds = "data_scientist/"
```


```python
jobs = pd.read_csv(root_dir+"jobs.csv")
jobs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>site</th>
      <th>job_url</th>
      <th>job_url_direct</th>
      <th>title</th>
      <th>company</th>
      <th>location</th>
      <th>area</th>
      <th>date_posted</th>
      <th>job_type</th>
      <th>...</th>
      <th>company_addresses</th>
      <th>company_num_employees</th>
      <th>company_revenue</th>
      <th>company_description</th>
      <th>skills</th>
      <th>experience_range</th>
      <th>company_rating</th>
      <th>company_reviews_count</th>
      <th>vacancy_count</th>
      <th>work_from_home_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>li-4323516725</td>
      <td>linkedin</td>
      <td>https://www.linkedin.com/jobs/view/4323516725</td>
      <td>NaN</td>
      <td>Junior Software Engineer</td>
      <td>Brooksource</td>
      <td>Birmingham, AL</td>
      <td>Birmingham, AL</td>
      <td>2025-11-21</td>
      <td>fulltime</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>li-4333159993</td>
      <td>linkedin</td>
      <td>https://www.linkedin.com/jobs/view/4333159993</td>
      <td>https://shipt.wd1.myworkdayjobs.com/Shipt_Exte...</td>
      <td>Senior Engineer</td>
      <td>Shipt</td>
      <td>Birmingham, AL</td>
      <td>Birmingham, AL</td>
      <td>2025-11-21</td>
      <td>fulltime</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>li-4256077277</td>
      <td>linkedin</td>
      <td>https://www.linkedin.com/jobs/view/4256077277</td>
      <td>https://industrycareers-enercon.icims.com/jobs...</td>
      <td>Physical Security Engineer</td>
      <td>Enercon Services, Inc.</td>
      <td>Birmingham, AL</td>
      <td>Birmingham, AL</td>
      <td>2025-11-21</td>
      <td>fulltime</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>li-4338291784</td>
      <td>linkedin</td>
      <td>https://www.linkedin.com/jobs/view/4338291784</td>
      <td>NaN</td>
      <td>Distinguished Engineer - AI Infrastructure Arc...</td>
      <td>Cisco</td>
      <td>Birmingham, AL</td>
      <td>Birmingham, AL</td>
      <td>2025-11-21</td>
      <td>fulltime</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>li-4302046287</td>
      <td>linkedin</td>
      <td>https://www.linkedin.com/jobs/view/4302046287</td>
      <td>NaN</td>
      <td>Information Services Quality Assurance Interns...</td>
      <td>Altec</td>
      <td>Birmingham, AL</td>
      <td>Birmingham, AL</td>
      <td>2025-11-21</td>
      <td>internship</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>




```python
jobs.shape
```




    (3060, 35)



I always use `{dfname}.info()` after `{dfname}.head()`. `{dfname}.head()` gives me a general sense, but `{dfname}.info()` gives me signals on which of the fields are actually useful. In this instance, most of the fields that I expect to have values, like `title`, `company`, `description`, `company_industry` are all filled with values. A couple is lacking some values like `location` and `date_posted`. Since I didn't build the scraper, I don't know why they're lacking. It could be crucial, but since the amount of values lacking are not a lot and we supposedly already know the range, it seems fine for me to leave it as is. Sometimes null values like this could reveal crucial problems in the system if this analysis is for business purposes, but that's not the case here. 

The other thing is a few fields don't have all the values but some, like `job_url_direct`, `emails`, `salary_source`, `interval`, `min_amount`, `max_amount` and `currency`. We won't be analyzing all of these fields in this notebook, but it's interesting to see, for example, 898 out of the 3060 jobs have direct link to their job postings and 573 out of the 3060 jobs have emails. Only 143 out of all jobs have salary ranges, which we'll do some light analysis on later.


```python
jobs.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3060 entries, 0 to 3059
    Data columns (total 35 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   id                     3060 non-null   object 
     1   site                   3060 non-null   object 
     2   job_url                3060 non-null   object 
     3   job_url_direct         898 non-null    object 
     4   title                  3060 non-null   object 
     5   company                3060 non-null   object 
     6   location               2903 non-null   object 
     7   area                   3060 non-null   object 
     8   date_posted            2607 non-null   object 
     9   job_type               3060 non-null   object 
     10  salary_source          143 non-null    object 
     11  interval               143 non-null    object 
     12  min_amount             143 non-null    float64
     13  max_amount             143 non-null    float64
     14  currency               143 non-null    object 
     15  is_remote              3060 non-null   bool   
     16  job_level              3060 non-null   object 
     17  job_function           3059 non-null   object 
     18  listing_type           0 non-null      float64
     19  emails                 573 non-null    object 
     20  description            3060 non-null   object 
     21  company_industry       3060 non-null   object 
     22  company_url            3060 non-null   object 
     23  company_logo           3060 non-null   object 
     24  company_url_direct     0 non-null      float64
     25  company_addresses      0 non-null      float64
     26  company_num_employees  0 non-null      float64
     27  company_revenue        0 non-null      float64
     28  company_description    0 non-null      float64
     29  skills                 0 non-null      float64
     30  experience_range       0 non-null      float64
     31  company_rating         0 non-null      float64
     32  company_reviews_count  0 non-null      float64
     33  vacancy_count          0 non-null      float64
     34  work_from_home_type    0 non-null      float64
    dtypes: bool(1), float64(14), object(20)
    memory usage: 815.9+ KB



```python
jobs.columns
```




    Index(['id', 'site', 'job_url', 'job_url_direct', 'title', 'company',
           'location', 'area', 'date_posted', 'job_type', 'salary_source',
           'interval', 'min_amount', 'max_amount', 'currency', 'is_remote',
           'job_level', 'job_function', 'listing_type', 'emails', 'description',
           'company_industry', 'company_url', 'company_logo', 'company_url_direct',
           'company_addresses', 'company_num_employees', 'company_revenue',
           'company_description', 'skills', 'experience_range', 'company_rating',
           'company_reviews_count', 'vacancy_count', 'work_from_home_type'],
          dtype='object')



### 📌 Map of jobs counts of cities across the US


```python
import plotly.express as px
import plotly.io as pio
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time
```

Convert each (city, state) to latitude and longitude to locate in the US map.


```python
'''
area_counts = jobs.groupby("area").size().reset_index(name="count")

geolocator = Nominatim(user_agent="job_mapper", timeout=10)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

lat_list = []
lon_list = []

for area in area_counts["area"]:
    loc = geocode(f"{area}, USA")
    if loc:
        lat_list.append(loc.latitude)
        lon_list.append(loc.longitude)
    else:
        lat_list.append(None)
        lon_list.append(None)
    time.sleep(1)  # required to avoid rate-limit

area_counts["lat"] = lat_list
area_counts["lon"] = lon_list
'''
```




    '\narea_counts = jobs.groupby("area").size().reset_index(name="count")\n\ngeolocator = Nominatim(user_agent="job_mapper", timeout=10)\ngeocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)\n\nlat_list = []\nlon_list = []\n\nfor area in area_counts["area"]:\n    loc = geocode(f"{area}, USA")\n    if loc:\n        lat_list.append(loc.latitude)\n        lon_list.append(loc.longitude)\n    else:\n        lat_list.append(None)\n        lon_list.append(None)\n    time.sleep(1)  # required to avoid rate-limit\n\narea_counts["lat"] = lat_list\narea_counts["lon"] = lon_list\n'




```python
area_counts = pd.read_csv(root_dir+"area2latlon.csv")

fig = px.scatter_geo(
    area_counts,
    lat="lat",
    lon="lon",
    size="count",
    hover_name="area",
    hover_data={"count": True},
    size_max=50,
    opacity=0.7,
    color="count",
    color_continuous_scale="Viridis",
    projection="albers usa",
)

# Balanced background and black state lines
fig.update_layout(
    title="Jobs Counts By City Across The US",
    title_font_size=20,
    title_font_color="#555555",   # grey title
    geo=dict(
        scope="usa",
        showland=True,
        landcolor="#e0e0e0",        # medium-light gray
        lakecolor="#d0e8ff",        # pale blue lakes
        showlakes=True,
        showcountries=True,
        countrycolor="gray",
        showcoastlines=False,
        bgcolor="#f5f5f5",           # soft gray background
        showsubunits=True,
        subunitcolor="gray",         # black state lines
    ),
    paper_bgcolor="#f5f5f5",         # soft gray outside map
    font=dict(color="black"),
    margin={"r":0,"t":50,"l":0,"b":0},
    height=500,
    width=1000,
)

# Optional: add city labels
fig.update_traces(text=area_counts["area"], textposition="top center")

fig.show()
```




```python
fig = px.treemap(
    area_counts,
    path=["area"],          # Each rectangle = area
    values="count",         # Rectangle size = job count
    color="count",          # Color by job count
    color_continuous_scale="Viridis",  # Pretty gradient
    hover_data={"count": True},
)

# Layout improvements
fig.update_layout(
    title="Ratio of Jobs Counts Per City Across The US",
    title_font_size=22,
    margin=dict(t=50, l=25, r=25, b=25),
    height=500,
    width=1000,
)

# Optional: customize text
fig.update_traces(
    texttemplate="%{label}\n%{value}",  # show area + count inside each rectangle
    textfont_size=14
)

fig.show()
```


    
![png](backend_eng_review_files/backend_eng_review_14_0.png)
    


Each city's job data is manually scraped using [an existing scraping API](https://github.com/speedyapply/JobSpy), so it's black box why the scraper found the resulting number of jobs. The number is correlated with our impression of which city has more tech jobs, but there's no way, for example, Boston, SF and Denver all have exactly 140 jobs at the time it's being scraped. 

In any way, there're more jobs in commonly-known tech hubs like San Francisco and San Jose, CA (= 140). But there are also a lot of backend engineering jobs in big cities like Baltimore, Washington DC and Chicago. It might be expected that Southern state have fewer jobs, but it's good to see that these following cities also have a decent amount of jobs:

- Atlanta, GA
- Raleigh, NC
- Charlotte, NC
- Detroit, MI (80)
- Minneapolis, MN (75)
- Cincinnati, OH (70)
- Phoenix, AZ (70)
- Salt Lake City, UT (70)
- Hartford, CT (70)
- Pittsburgh, PA (69)

Especially Atlanta, GA has 120 jobs, Raleigh and Charlotte from North Carolina have 90 and 89 jobs found respectively. For comparison, Hartford, CT also have 70 jobs and Hartford is one of the small cities that has a lot of very well-paid tech jobs right above the NY state.

Lastly, it seems from the US map of job counts by city that the majority of the jobs are from big tech cities. However, breaking it down in the second box/proportion graph, it's obvious that the number of jobs from big tech cities is clearly below the total number of jobs in the dataset. Moreover, there's a significant representation of cities from red states in the rest of the dataset, demonstrated by the city names in most of the boxes on the right side of the plot. 

## 💵 Salary

Since `salary` field only has 42 values, it's convenient to analyze that and just get it out of the way.


```python
salary_source = jobs.salary_source.dropna().tolist()
intervals = jobs.interval.dropna().tolist()
min_amount = jobs.min_amount.dropna().tolist()
max_amount = jobs.max_amount.dropna().tolist()
print("First value of each related field \n"+
f"salary_source: \"{salary_source[0]}\" \ninterval: {intervals[0]} \nmin_amount: {min_amount[0]} \nmax_amount: {max_amount[0]}")
```

    First value of each related field 
    salary_source: "description" 
    interval: yearly 
    min_amount: 90000.0 
    max_amount: 110000.0


From experience `salary_source` and `interval` would be the same for each of the 42 row. Just for scientific spirits, let's verify that.


```python
salary_source_unique = jobs.salary_source.dropna().unique().tolist()
intervals_unique = jobs.interval.dropna().unique().tolist()
print(f"Salary source(s): {salary_source_unique} \nIntervals: {intervals_unique}")
```

    Salary source(s): ['description'] 
    Intervals: ['yearly', 'hourly']


Turned out it's good we checked. When inspecting min salary amount and max salary amount, we'll check these seperately.


```python
min_amount_yearly = jobs[jobs.interval == 'yearly'].min_amount.dropna().tolist()
max_amount_yearly = jobs[jobs.interval == 'yearly'].max_amount.dropna().tolist()
min_amount_hourly = jobs[jobs.interval == 'hourly'].min_amount.dropna().tolist()
max_amount_hourly = jobs[jobs.interval == 'hourly'].max_amount.dropna().tolist()
print(f"# of yearly salary instances: {len(min_amount_yearly)} \n# of hourly salary instance(s): {len(min_amount_hourly)}")
```

    # of yearly salary instances: 140 
    # of hourly salary instance(s): 3



```python
for i in range(3):
    print(f"Hourly rate range {i+1}: ${min_amount_hourly[i]}/hr - ${max_amount_hourly[i]}/hr")
```

    Hourly rate range 1: $30.0/hr - $33.0/hr
    Hourly rate range 2: $75.0/hr - $80.0/hr
    Hourly rate range 3: $40.0/hr - $45.0/hr


We can translate hourly rates to yearly rates assuming the job works 40hr/week and 4 weeks/month.


```python
for i in range(3):
    min_amount_yearly.append(min_amount_hourly[i]*40*4*12)
    max_amount_yearly.append(max_amount_hourly[i]*40*4*12)
```


```python
print(f"Lower end of minimal amount of salary: ${min(min_amount_yearly)/1000}k \n" +
      f"Higher end of minimal amount of salary: ${max(min_amount_yearly)/1000}k")
```

    Lower end of minimal amount of salary: $50.0k 
    Higher end of minimal amount of salary: $300.0k



```python
print(f"Lower end of maximal amount of salary: ${min(max_amount_yearly)/1000}k \n" +
      f"Higher end of maximal amount of salary: ${max(max_amount_yearly)/1000}k")
```

    Lower end of maximal amount of salary: $63.36k 
    Higher end of maximal amount of salary: $405.0k


### Distribution of Salary Range


```python
import matplotlib.pyplot as plt
import logging
logging.getLogger("matplotlib.font_manager").disabled = True
```


```python
y_min = [1] * len(min_amount_yearly)
y_max = [2] * len(max_amount_yearly)

plt.xkcd()

plt.figure(figsize=(9, 4))
plt.scatter([x/1000 for x in min_amount_yearly], y_min, s=200, color='#FFB347', label='List1', marker='o', alpha=0.7)
plt.scatter([x/1000 for x in max_amount_yearly], y_max, s=200, color='#6A5ACD', label='List2', marker='o', alpha=0.7)

# Customize axes
plt.xticks(range(0, int(max(max_amount_yearly)/1000), 10), rotation=60)
plt.yticks([1,2], ['min salary', 'max salary'], rotation=45)
plt.xlabel('salary (k/yr)')
y_min = 0.5
y_max = 2.45
plt.ylim(y_min, y_max)
plt.title('Backend engineer salary range distribution')
plt.grid(axis='x', linestyle=':', linewidth=1.65, color='#333333', alpha=0.7)

plt.show()
```


    
![png](backend_eng_review_files/backend_eng_review_29_0.png)
    


It feels like the lower end of most salary ranges between \\$50k - \\$210k and the higher end of most salary ranges from \\$65k to \\$300k. By my own job browsing experience, this is pretty accurate. Compared to most tech jobs, this actually seems a bit low (quite low). If you search for data scientist or ML Eng jobs, the higher end could easily go up to around 600k to 800k. The following analysis will show,  however, **why in my opinion, backend engineering is a great first job for job seekers in this market and will likely open up further opportunities.**

**P.S.** If you're wondering why the graph is all doodles it's because of this magical command `plt.xkcd()`. You don't have to download extra packages because it's included in `matplotlib` but you do have to suppress the font manager warnings using `logging.getLogger("matplotlib.font_manager").disabled = True`.


```python
import plotly.graph_objects as go
```


```python
min_amount_yearly = jobs[jobs.interval == 'yearly'].min_amount.dropna()
max_amount_yearly = jobs[jobs.interval == 'yearly'].max_amount.dropna()
min_amount_hourly = jobs[jobs.interval == 'hourly'].min_amount.dropna()
max_amount_hourly = jobs[jobs.interval == 'hourly'].max_amount.dropna()
min_amount_hourly = min_amount_hourly * 40 * 4 * 12
max_amount_hourly = max_amount_hourly * 40 * 4 * 12
min_amount_yearly = pd.concat([min_amount_yearly, min_amount_hourly], ignore_index=True)
max_amount_yearly = pd.concat([max_amount_yearly, max_amount_hourly], ignore_index=True)

indices = jobs.min_amount.dropna().index.tolist()
areas_with_valid_rates = jobs.loc[indices].area.tolist()

data = {"area": areas_with_valid_rates,
       "min_amount": min_amount_yearly.tolist(),
       "max_amount": max_amount_yearly.tolist()}

df_salary_by_area = pd.DataFrame(data)
df_area_min_salary_median = (
    df_salary_by_area.groupby("area")
    .agg(
        min_amount_median=("min_amount", "median"),
        count=("min_amount", "count")   # count rows per area
    )
    .reset_index()
)

# ---- Median max salary + count ----
df_area_max_salary_median = (
    df_salary_by_area.groupby("area")
    .agg(
        max_amount_median=("max_amount", "median"),
        count=("max_amount", "count")   # same count (based on max_amount)
    )
    .reset_index()
)
```


```python
area_counts = pd.read_csv(root_dir+"area2latlon.csv")
```


```python
df_area_min_salary_median = df_area_min_salary_median.merge(
    area_counts[['area', 'lat', 'lon']],
    on='area',
    how='left'
)

# Get sets of areas
all_areas = set(jobs["area"])
median_areas = set(df_area_min_salary_median["area"])

# Areas present in jobs but missing in median
missing_areas = list(all_areas - median_areas)

# Subset lat/lon for missing areas
missing_coords = area_counts[area_counts["area"].isin(missing_areas)][['area', 'lat', 'lon']]
```


```python
# Base bubble map for median salaries
fig = px.scatter_geo(
    df_area_min_salary_median,
    lat="lat",
    lon="lon",
    size="min_amount_median",
    color="min_amount_median",
    hover_name="area",
    hover_data={
        "min_amount_median": True,
        "count": True,       # <-- add your jobs column here
        "lat": False,        # hide lat/lon if you want
        "lon": False
    },
    size_max=50,
    opacity=0.7,
    color_continuous_scale="Viridis",
    projection="albers usa"
)

# Add missing areas as 'X' markers with white background for text
fig.add_trace(
    go.Scattergeo(
        lat=missing_coords['lat'],
        lon=missing_coords['lon'],
        mode='markers+text',
        marker=dict(symbol='x', size=12, color='red'),
        text=missing_coords['area'],
        textposition="top center",
        textfont=dict(
            size=12,
            color='#333333',
        ),
        name='Missing Areas',
    )
)

# Mid-tone map style with black state lines
fig.update_layout(
    title="Median Min Yearly Amount by City (with Missing Areas Marked)",
    title_font_size=22,
    title_font_color="#555555",
    geo=dict(
        scope="usa",
        showland=True,
        landcolor="#e0e0e0",
        lakecolor="#d0e8ff",
        showlakes=True,
        showcountries=True,
        countrycolor="gray",
        showcoastlines=False,
        bgcolor="#f5f5f5",
        showsubunits=True,
        subunitcolor="black",
    ),
    paper_bgcolor="#f5f5f5",
    font=dict(color="black"),
    margin={"r":0,"t":50,"l":0,"b":0},
    height=600,
    width=1000,
)

fig.update_layout(
    legend=dict(
        x=0.02,           # move legend left
        y=0.98,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="black",
        borderwidth=1
    )
)

fig.show()
```


    
![png](backend_eng_review_files/backend_eng_review_35_0.png)
    



```python
df_area_max_salary_median = df_area_max_salary_median.merge(
    area_counts[['area', 'lat', 'lon']],
    on='area',
    how='left'
)
```


```python
# Base bubble map for median salaries
fig = px.scatter_geo(
    df_area_max_salary_median,
    lat="lat",
    lon="lon",
    size="max_amount_median",
    color="max_amount_median",
    hover_name="area",
    hover_data={
        "max_amount_median": True,
        "count": True,       # <-- add your jobs column here
        "lat": False,        # hide lat/lon if you want
        "lon": False
    },
    size_max=50,
    opacity=0.7,
    color_continuous_scale="Viridis",
    projection="albers usa"
)

# Add missing areas as 'X' markers with white background for text
fig.add_trace(
    go.Scattergeo(
        lat=missing_coords['lat'],
        lon=missing_coords['lon'],
        mode='markers+text',
        marker=dict(symbol='x', size=12, color='red'),
        text=missing_coords['area'],
        textposition="top center",
        textfont=dict(
            size=12,
            color='#333333',
        ),
        name='Missing Areas',
    )
)

# Mid-tone map style with black state lines
fig.update_layout(
    title="Median Max Yearly Amount by City (with Missing Areas Marked)",
    title_font_size=22,
    title_font_color="#555555",
    geo=dict(
        scope="usa",
        showland=True,
        landcolor="#e0e0e0",
        lakecolor="#d0e8ff",
        showlakes=True,
        showcountries=True,
        countrycolor="gray",
        showcoastlines=False,
        bgcolor="#f5f5f5",
        showsubunits=True,
        subunitcolor="black",
    ),
    paper_bgcolor="#f5f5f5",
    font=dict(color="black"),
    margin={"r":0,"t":50,"l":0,"b":0},
    height=600,
    width=1000,
)

fig.update_layout(
    legend=dict(
        x=0.02,           # move legend left
        y=0.98,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="black",
        borderwidth=1
    )
)

fig.show()
```


    
![png](backend_eng_review_files/backend_eng_review_37_0.png)
    


The first thing to notice is a few cities you wouldn't normally expect to have high median salaries have higher median salaries than big tech cities like East coast cities and even San Francisco and San Jose. If you hover on the bubbles, the labels tell you how many job in that city had salary stated in the listing. It's clear from the job count labels that most of these cities only have 1 to 3 jobs that have salary labels, which means those are the only salaries to take median values from. What it means is that at least the median salary of these cities could not be fully trusted. Another possible interpretation could be that only companies who can offer good salary listed the number, which sound much more possible but is hard to verify for sure.

The couple other things to notice is that San Diego, CA has a bit more jobs (5) that have salary listing and higher salary median, \\$172k/yr - \\$258k/yr. Next, the following cities have a couple salary listings that showed that they actually don't pay that well (diabolically).

- Salt Lake City, UT (\\$101k/yr - \\$108.5k/yr)
- Milwaukee, WI (\\$77k/yr - \\$86k/yr)
- Detroit, MI (\\$80k/yr - \\$110k/yr)
- Indianapolis, IN (\\$70k/yr - \\$75k/yr)
- Birmingham, AL (\\$90k/yr - \\$110k/yr)

The fact that these cities only have a couple salary listings are already suspicious. Moreover, the median salaries in most cities on the graph is \\$130k/yr - \\$195k/yr if you hover over the darker green bubbles. So having only a couple listings that ranges from around \\$80k/yr to barely reaching \\$110k/yr demonstrates that these cities might not have the most well-paid jobs. 

Lastly, the cities that don't have jobs with salary listed are a mix of a couple big cities or cities that I know have good tech jobs like Hartford, CT, and major cities from red states. I'd not draw any definite conclusions from it.

## 🏅 Job Level

Data scientist jobs collected from a few sampled big tech cities are used here in comparison to the backend engineering job dataset.


```python
jobs_ds = pd.read_csv(root_dir+"jobs_ds.csv")
print(f"The data scientist dataset has {jobs_ds.shape[0]} jobs. ")
```

    The data scientist dataset has 834 jobs. 



```python
# Prepare data
levels_backend = jobs.job_level
levels_freq_backend = levels_backend.value_counts()

levels_ds = jobs_ds.job_level
levels_freq_ds = levels_ds.value_counts()

# Create subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

# --- Backend Engineering Plot ---
levels_freq_backend.plot(kind='bar', ax=axes[0])
axes[0].set_xlabel("Job Levels")
axes[0].set_ylabel("Frequency")
axes[0].set_xticklabels(levels_freq_backend.index, rotation=45)
axes[0].grid(axis='y', linestyle=':', linewidth=1.65, color='#333333', alpha=0.7)

# Add numerical labels
for i, value in enumerate(levels_freq_backend):
    axes[0].text(i, value + 0.05, str(value), ha='center', va='bottom', fontsize=12)

axes[0].set_title("Frequently Demanded Job Levels (Backend Engineering)", fontsize=14)

# --- Data Scientist Plot ---
levels_freq_ds.plot(kind='bar', ax=axes[1], color='orange')  # optional color change
axes[1].set_xlabel("Job Levels")
axes[1].set_ylabel("Frequency")
axes[1].set_xticklabels(levels_freq_ds.index, rotation=45)
axes[1].grid(axis='y', linestyle=':', linewidth=1.65, color='#333333', alpha=0.7)

# Add numerical labels
for i, value in enumerate(levels_freq_ds):
    axes[1].text(i, value + 0.05, str(value), ha='center', va='bottom', fontsize=12)

axes[1].set_title("Frequently Demanded Job Levels (Data Scientist)", fontsize=14)

plt.tight_layout()
plt.show()
```


    
![png](backend_eng_review_files/backend_eng_review_41_0.png)
    


In agreement with the trends in the tech industry, there are more mid to senior level jobs than entry-level jobs. In comparison to the most popular job title "Data Scientist", of which in this sampled dataset from major tech cities half of which the job is entry-level, about 36.4% of backend engineering jobs across the US is entry-level jobs, which is pretty good. Moreover, the definition of mid-senior level jobs could be different for backend engineers compared to data scientists. Because data scientists require various understanding of machine learning or deep learning as well as requirements of a masters or even PhD degree, it's much more difficult to get to mid-senior level. In comparison, it's much easier to get to mid-senior level as a backend engineer. 


```python
mid_senior_jobs = jobs[jobs["job_level"] == "mid-senior level"]
entry_jobs = jobs[jobs["job_level"] == "entry level"]

mid_senior_counts = mid_senior_jobs.groupby("area").size().reset_index(name="job_count")
entry_counts = entry_jobs.groupby("area").size().reset_index(name="job_count")

# Merge lat/lon
mid_senior_counts = mid_senior_counts.merge(area_counts[['area','lat','lon']], on='area', how='left')
entry_counts = entry_counts.merge(area_counts[['area','lat','lon']], on='area', how='left')
```


```python
# Entry Level Map
fig2 = px.scatter_geo(
    entry_counts,
    lat="lat",
    lon="lon",
    size="job_count",
    color="job_count",
    hover_name="area",
    size_max=50,
    opacity=0.7,
    color_continuous_scale="Viridis",
    projection="albers usa",
    title="Entry Level Job Count by City"
)

fig2.update_layout(
    geo=dict(
        scope="usa",
        showland=True,
        landcolor="#e0e0e0",
        lakecolor="#d0e8ff",
        showlakes=True,
        showcountries=True,
        countrycolor="gray",
        showcoastlines=False,
        bgcolor="#f5f5f5",
        showsubunits=True,
        subunitcolor="black",
    ),
    paper_bgcolor="#f5f5f5",
    height=600,
    width=1000,
)

fig2.show()

# Mid-Senior Level Map
fig1 = px.scatter_geo(
    mid_senior_counts,
    lat="lat",
    lon="lon",
    size="job_count",
    color="job_count",
    hover_name="area",
    size_max=50,
    opacity=0.7,
    color_continuous_scale="Viridis",
    projection="albers usa",
    title="Mid-Senior Level Job Count by City"
)

fig1.update_layout(
    geo=dict(
        scope="usa",
        showland=True,
        landcolor="#e0e0e0",
        lakecolor="#d0e8ff",
        showlakes=True,
        showcountries=True,
        countrycolor="gray",
        showcoastlines=False,
        bgcolor="#f5f5f5",
        showsubunits=True,
        subunitcolor="black",
    ),
    paper_bgcolor="#f5f5f5",
    height=600,
    width=1000,
)

fig1.show()
```


    
![png](backend_eng_review_files/backend_eng_review_44_0.png)
    



    
![png](backend_eng_review_files/backend_eng_review_44_1.png)
    


It seems like the following cities have more entry-level jobs, ranked from the most jobs to fewer jobs:

- Seattle, CA (38)
- Sacramento, CA (33)
- Denver, CO (32)
- San Jose, CA (30)
- San Francisco, CA (28)
- Baltimore, MD (28)
- Boston, MA (25)
- Washington D.C., D.C. (25)

And the following cities have more mid-senior level jobs, a bit more than 90 jobs: New York, NY / Baltimore, MD / Washington D.C., D.C. / Boston, MA / Denver, CO. Atlanta and Chicago each have around 70 jobs, while cities in the West Coast actually have a bit less (around 60) of them. 

## 🏢 Companies


```python
from wordcloud import WordCloud
```


```python
company_counts = jobs['company'].value_counts()

# View top 10 most mentioned companies
top_companies = company_counts.head(25)
print(top_companies)
```

    company
    Epic                         136
    Jobs via Dice                 65
    Google                        47
    PwC                           35
    ExecutivePlacements.com       34
    KPMG US                       34
    Intuit                        25
    GEICO                         25
    EY                            23
    Qualcomm                      22
    Pearson                       22
    Booz Allen Hamilton           21
    Meta                          21
    Tata Consultancy Services     19
    Uline                         18
    Mindrift                      18
    Adobe                         18
    Motion Recruitment            17
    Affirm                        17
    Capital One                   16
    Lockheed Martin               16
    Deloitte                      16
    Scribd, Inc.                  15
    Boeing                        15
    Raytheon                      15
    Name: count, dtype: int64



```python
google = company_counts["Google"]
meta = company_counts["Meta"]
amazon = company_counts["Amazon"]
netflix = company_counts["Netflix"]
apple = 0
if "Apple" in company_counts.index: 
    apple = company_counts["Apple"]
print(f"# of FANNG jobs \nGoogle: {google}\nMeta: {meta} \nAmazon: {amazon} \nNetflix: {netflix} \nApple: {apple}")
```

    # of FANNG jobs 
    Google: 47
    Meta: 21 
    Amazon: 1 
    Netflix: 14 
    Apple: 0



```python
# List of companies to remove
remove_companies = ["Jobs via Dice", "ExecutivePlacements.com", "Motion Recruitment"]

# Filter out these companies from the frequency dictionary
company_freq_filtered = {company: count for company, count in company_counts.items() if company not in remove_companies}

# Generate the word cloud from the filtered dictionary
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='viridis'
).generate_from_frequencies(company_freq_filtered)

# Plot
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Mentioned Companies", fontsize=24, font="DejaVu Sans")
plt.show()
```


    
![png](backend_eng_review_files/backend_eng_review_50_0.png)
    


By getting the most mentioned companies in this dataset, we can see that Epic the electronic health record (EHR) system listed the most backend engineering jobs (136 jobs). While some suspicious job websites are up on the list, like Jobs via Dice, ExecutivePlacement.com and Motion Recruitment, most of the companies are well-known like Google, PwC, etc. Some of the other well-known company on the list are: KPMG US, EY, Intuit, PwC, Qualcomm, Tata Consultancy Services, Booz Allen Hamilton, Affirm, Pearson, GEICO, and companies with less backend engineering jobs like Adobe, IBM, Lockheed Martin, Deloitte, NVIDIA and Boeing. 

A quick check people like to do is to look for FAANG company jobs. FAANG are five big tech companies recognized in the industry, standing for Facebook (now Meta), Amazon, Apple, Netflix, and Alphabet (Google) respectively. Here we have Google: 47 jobs, Meta: 21 jobs, Amazon: 1 job and Netflix: 14 jobs. Apple has no job listed, just because they don't post ANY job on ANY job boards and only on their own career website, so it's normal that our dataset doesn't contain any job from Apple. Our job source is LinkedIn.

## 🏭 Industry


```python
industry = jobs.company_industry

# List of companies to remove
remove_companies = ["Jobs via Dice", "ExecutivePlacements.com", "Motion Recruitment"]

# Filter the jobs dataframe to exclude these companies
jobs_filtered = jobs[~jobs['company'].isin(remove_companies)]

industry_counts = jobs_filtered['company_industry'].value_counts()

# Get top 10 most frequent industries
top_industries = industry_counts.head(7)
print(top_industries)
```

    company_industry
    Software Development                                                                      486
    IT Services and IT Consulting                                                             353
    Financial Services                                                                        217
    Technology, Information and Internet                                                      151
    Defense and Space Manufacturing                                                           117
    Insurance                                                                                  79
    Software Development, Hospitals and Health Care, and Information Technology & Services     68
    Name: count, dtype: int64



```python
import matplotlib.pyplot as plt

# Convert top_industries to a dataframe for easier manipulation
top_industries_df = top_industries.reset_index()
top_industries_df.columns = ['industry', 'count']

# Rename the 7th industry (index 6) to "Hospitals and Health Care Software"
top_industries_df.loc[6, 'industry'] = "Hospitals and Health Care Software"

# Plot as bar chart
plt.figure(figsize=(10,6))
bars = plt.bar(top_industries_df['industry'], top_industries_df['count'])

# Add text labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,  # x-position
        height + 0.5,                     # y-position slightly above bar
        str(int(height)),                  # label text
        ha='center', va='bottom', fontsize=12
    )

plt.xticks(fontsize=12)
plt.xlabel("Industry")
plt.ylabel("Number of Job Postings")
plt.title("Top 7 Most Frequent Industries")
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.grid(axis='y', linestyle=':', linewidth=1.65, color='#333333', alpha=0.7)

plt.show()
```


    
![png](backend_eng_review_files/backend_eng_review_54_0.png)
    


Most jobs are from the software development, consulting or financial services industry. Other more specific industries are Defense and Space Manufacturing, Insurance and Healthcare Software. Some examples of software development industry companies could be Slack, Pearson and Scribd, Inc. Consulting companies are straightforward, like EY and PwC. Financial services companies include Affirm, Intuit and Capital One. 

## 🌐 Is Remote


```python
is_remote = jobs_filtered.is_remote
remote_counts = jobs['is_remote'].value_counts()
print(remote_counts)
```

    is_remote
    False    2322
    True      738
    Name: count, dtype: int64



```python
# Count remote vs non-remote jobs
remote_counts = jobs_filtered['is_remote'].value_counts()
labels = ['Non-Remote', 'Remote']

# Custom function to show both count and percentage
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        count = int(round(pct*total/100.0))
        return f"{count} ({pct:.1f}%)"
    return my_autopct

plt.figure(figsize=(6,6))
plt.pie(
    remote_counts,
    labels=labels,
    autopct=make_autopct(remote_counts),
    colors=['#FF9999','#66B3FF'],
    startangle=90,
    shadow=True
)
plt.title("Proportion of Remote vs Non-Remote Jobs", fontsize=14)
plt.show()

```


    
![png](backend_eng_review_files/backend_eng_review_58_0.png)
    



```python
# Filter remote jobs
remote_jobs = jobs_filtered[jobs_filtered['is_remote'] == True]

# Group by area/city and count jobs
remote_counts = remote_jobs.groupby("area").size().reset_index(name="job_count")

# Merge lat/lon for plotting (assuming area_counts dataframe exists)
remote_counts = remote_counts.merge(area_counts[['area','lat','lon']], on='area', how='left')
```


```python
fig = px.scatter_geo(
    remote_counts,
    lat="lat",
    lon="lon",
    size="job_count",
    color="job_count",
    hover_name="area",
    size_max=50,
    opacity=0.7,
    color_continuous_scale="Viridis",
    projection="albers usa",
    title="Remote Job Count by City"
)

# Map layout styling
fig.update_layout(
    geo=dict(
        scope="usa",
        showland=True,
        landcolor="#e0e0e0",
        lakecolor="#d0e8ff",
        showlakes=True,
        showcountries=True,
        countrycolor="gray",
        showcoastlines=False,
        bgcolor="#f5f5f5",
        showsubunits=True,
        subunitcolor="black",
    ),
    paper_bgcolor="#f5f5f5",
    height=600,
    width=1000,
)

fig.show()
```


    
![png](backend_eng_review_files/backend_eng_review_60_0.png)
    


After filtering out the jobs from sketchy recruiting website like Dice, we see that about a quarter of the jobs are Remote. Separating remote jobs by city, the map shows that Denver (50), Baltimore (43), Washington DC (36), Boston, New York and Chicago. All of them have more than 30 remote jobs. Atlanta and Raleigh don't have as much but still a decent amount (29 jobs and 26 jobs, respectively). 

Additionally, it's interesting from all of the above analysis, traditionally well-regarded big tech cities like San Francisco and Seattle are seldom mentioned. Aside from the fact that those California cities have well-paid jobs, it seems like **New York, Denver, Baltimore, Washington DC, Boston, Chicago, Atlanta and Raleigh** have comparably well-paid jobs that are entry, mid-senior level and remote. One more specific thing you can look for by yourself is which city have which industry. This is too detailed for this review, but the dataset of all jobs, as well as by city and the sampled data scientist dataset will be uploaded to [Kaggle](www.kaggle.com) and available for download in the beginning and end of the notebook. 

## 🇹 Description (years of experience, skills, etc.)

## Title

## Logo (optional - another time?)

### Appendix 1: data aggregation and a list of cities covered by the dataset

**Table 1.** A list of files and corresponding city, states and number of jobs found

| #  | file name               | city                     | number of jobs found |
|----|--------------------------|--------------------------|-------------------------|
| 1  | jobs_AL.csv             | Birmingham, AL           | 30  |
| 2  | jobs_ALMontgomery.csv   | Montgomery, AL           | 19  |
| 3  | jobs_AZPheonix.csv      | Phoenix, AZ              | 70  |
| 4  | jobs_AZTucson.csv       | Tucson, AZ               | 30  |
| 5  | jobs_Boston.csv         | Boston, MA               | 140 |
| 6  | jobs_CASD.csv           | San Diego, CA            | 90  |
| 7  | jobs_CASacramento.csv   | Sacramento, CA           | 140 |
| 8  | jobs_CO.csv             | Denver, CO               | 140 |
| 9  | jobs_CT.csv             | Hartford, CT             | 70  |
| 10 | jobs_DC.csv             | Washington D.C., D.C.    | 138 |
| 11 | jobs_GA.csv             | Atlanta, GA              | 120 |
| 12 | jobs_GASavannah.csv     | Savannah, GA             | 4   |
| 13 | jobs_IA.csv             | Des Moines, IA           | 38  |
| 14 | jobs_IN.csv             | Indianapolis, IN         | 60  |
| 15 | jobs_INChicago.csv      | Chicago, IN              | 119 |
| 16 | jobs_KY.csv             | Louisville, KY           | 33  |
| 17 | jobs_LA.csv             | New Orleans, LA          | 21  |
| 18 | jobs_MD.csv             | Baltimore, MD            | 139 |
| 19 | jobs_MI.csv             | Detroit, MI              | 80  |
| 20 | jobs_MN.csv             | Minneapolis, MN          | 75  |
| 21 | jobs_MO.csv             | St Louis, MO             | 60  |
| 22 | jobs_MOKC.csv           | Kansas City, MO          | 56  |
| 23 | jobs_MS.csv             | Jackson, MS              | 24  |
| 24 | jobs_NC.csv             | Raleigh, NC              | 90  |
| 25 | jobs_NCCharlotte.csv    | Charlotte, NC            | 89  |
| 26 | jobs_NM.csv             | Santa Fe, NM             | 14  |
| 27 | jobs_NMAlbuquerque.csv  | Albuquerque, NM          | 28  |
| 28 | jobs_NY.csv             | New York, NY             | 140 |
| 29 | jobs_OH.csv             | Cincinnati, OH           | 70  |
| 30 | jobs_OHCleveland.csv    | Cleveland, OH            | 50  |
| 31 | jobs_OK.csv             | Oklahoma City, OK        | 30  |
| 32 | jobs_OR.csv             | Portland, OR             | 60  |
| 33 | jobs_PA.csv             | Pittsburgh, PA           | 69  |
| 34 | jobs_SC.csv             | Charleston, SC           | 30  |
| 35 | jobs_SCColumbia.csv     | Columbia, SC             | 30  |
| 36 | jobs_SF.csv             | San Francisco, CA        | 140 |
| 37 | jobs_SanJose.csv        | San Jose, CA             | 140 |
| 38 | jobs_Seattle.csv        | Seattle, CA              | 140 |
| 39 | jobs_TN.csv             | Memphis, TN              | 27  |
| 40 | jobs_UT.csv             | Salt Lake City, UT       | 70  |
| 41 | jobs_VA.csv             | Richmond, VA             | 60  |
| 42 | jobs_WI.csv             | Milwaukee, WI            | 57  |
| 43 | jobs_WIMadison.csv      | Madison, WI              | 30  |
|    | **TOTAL**               | —                        | **3060** |

### Aggregating backend engineering jobs


```python
root_dir = "data/"
job_dir_backend_eng = "backend_engineer/"
job_dir_ds = "data_scientist/"
files = os.listdir(root_dir + job_dir_backend_eng)
print(files), len(files)
```

    ['jobs_PA.csv', 'jobs_IA.csv', 'jobs_Boston.csv', 'jobs_DC.csv', '.DS_Store', 'jobs_NMAlbuquerque.csv', 'jobs_Seattle.csv', 'jobs_KY.csv', 'jobs_MI.csv', 'jobs_CT.csv', 'jobs_NCCharlotte.csv', 'jobs_VA.csv', 'jobs_SanJose.csv', 'jobs_AL.csv', 'jobs_NC.csv', 'jobs_MO.csv', 'jobs_MN.csv', 'jobs_AZTucson.csv', 'jobs_TN.csv', 'jobs_OR.csv', 'jobs_CASD.csv', 'jobs_OH.csv', 'jobs_NM.csv', 'jobs_GASavannah.csv', 'jobs_INChicago.csv', 'jobs_OK.csv', 'jobs_NY.csv', 'jobs_ALMontgomery.csv', 'jobs_MOKC.csv', 'jobs_WI.csv', 'jobs_WIMadison.csv', 'jobs_OHCleveland.csv', 'jobs_UT.csv', 'jobs_LA.csv', 'jobs_MS.csv', 'jobs_MD.csv', 'jobs_CO.csv', 'jobs_GA.csv', 'jobs_CASacramento.csv', 'jobs_SF.csv', 'jobs_SCColumbia.csv', 'jobs_AZPheonix.csv', 'jobs_SC.csv', 'jobs_IN.csv']





    (None, 44)




```python
# Dictionary mapping filenames to city (area)
csv2area = {
    "jobs_AL.csv": "Birmingham, AL",
    "jobs_ALMontgomery.csv": "Montgomery, AL",
    "jobs_AZPheonix.csv": "Phoenix, AZ",
    "jobs_AZTucson.csv": "Tucson, AZ",
    "jobs_Boston.csv": "Boston, MA",
    "jobs_CASD.csv": "San Diego, CA",
    "jobs_CASacramento.csv": "Sacramento, CA",
    "jobs_CO.csv": "Denver, CO",
    "jobs_CT.csv": "Hartford, CT",
    "jobs_DC.csv": "Washington D.C., D.C.",
    "jobs_GA.csv": "Atlanta, GA",
    "jobs_GASavannah.csv": "Savannah, GA",
    "jobs_IA.csv": "Des Moines, IA",
    "jobs_IN.csv": "Indianapolis, IN",
    "jobs_INChicago.csv": "Chicago, IN",
    "jobs_KY.csv": "Louisville, KY",
    "jobs_LA.csv": "New Orleans, LA",
    "jobs_MD.csv": "Baltimore, MD",
    "jobs_MI.csv": "Detroit, MI",
    "jobs_MN.csv": "Minneapolis, MN",
    "jobs_MO.csv": "St Louis, MO",
    "jobs_MOKC.csv": "Kansas City, MO",
    "jobs_MS.csv": "Jackson, MS",
    "jobs_NC.csv": "Raleigh, NC",
    "jobs_NCCharlotte.csv": "Charlotte, NC",
    "jobs_NM.csv": "Santa Fe, NM",
    "jobs_NMAlbuquerque.csv": "Albuquerque, NM",
    "jobs_NY.csv": "New York, NY",
    "jobs_OH.csv": "Cincinnati, OH",
    "jobs_OHCleveland.csv": "Cleveland, OH",
    "jobs_OK.csv": "Oklahoma City, OK",
    "jobs_OR.csv": "Portland, OR",
    "jobs_PA.csv": "Pittsburgh, PA",
    "jobs_SC.csv": "North Charleston, SC",
    "jobs_SCColumbia.csv": "Columbia, SC",
    "jobs_SF.csv": "San Francisco, CA",
    "jobs_SanJose.csv": "San Jose, CA",
    "jobs_Seattle.csv": "Seattle, CA",
    "jobs_TN.csv": "Memphis, TN",
    "jobs_UT.csv": "Salt Lake City, UT",
    "jobs_VA.csv": "Richmond, VA",
    "jobs_WI.csv": "Milwaukee, WI",
    "jobs_WIMadison.csv": "Madison, WI",
}

# List of filenames to iterate through
# Example: files = list(csv2area.keys())
files = list(csv2area.keys())

all_jobs = []   # to collect all dataframes

for file in files:
    jobs = pd.read_csv(root_dir+job_dir_backend_eng+file)          # load CSV
    jobs["area"] = csv2area[file]     # assign area
    if "location" in jobs.columns:
        cols = list(jobs.columns)
        cols.remove("area")
        loc_index = cols.index("location") + 1
        cols.insert(loc_index, "area")
        jobs = jobs[cols]
    all_jobs.append(jobs)             # store df

# Stack/aggregate all dataframes
df = pd.concat(all_jobs, ignore_index=True)

# Save aggregated file
df.to_csv("jobs.csv", index=False)
```


```python
jobs = pd.read_csv(root_dir+"jobs.csv")
```


```python
jobs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>site</th>
      <th>job_url</th>
      <th>job_url_direct</th>
      <th>title</th>
      <th>company</th>
      <th>location</th>
      <th>area</th>
      <th>date_posted</th>
      <th>job_type</th>
      <th>...</th>
      <th>company_addresses</th>
      <th>company_num_employees</th>
      <th>company_revenue</th>
      <th>company_description</th>
      <th>skills</th>
      <th>experience_range</th>
      <th>company_rating</th>
      <th>company_reviews_count</th>
      <th>vacancy_count</th>
      <th>work_from_home_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>li-4323516725</td>
      <td>linkedin</td>
      <td>https://www.linkedin.com/jobs/view/4323516725</td>
      <td>NaN</td>
      <td>Junior Software Engineer</td>
      <td>Brooksource</td>
      <td>Birmingham, AL</td>
      <td>Birmingham, AL</td>
      <td>2025-11-21</td>
      <td>fulltime</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>li-4333159993</td>
      <td>linkedin</td>
      <td>https://www.linkedin.com/jobs/view/4333159993</td>
      <td>https://shipt.wd1.myworkdayjobs.com/Shipt_Exte...</td>
      <td>Senior Engineer</td>
      <td>Shipt</td>
      <td>Birmingham, AL</td>
      <td>Birmingham, AL</td>
      <td>2025-11-21</td>
      <td>fulltime</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>li-4256077277</td>
      <td>linkedin</td>
      <td>https://www.linkedin.com/jobs/view/4256077277</td>
      <td>https://industrycareers-enercon.icims.com/jobs...</td>
      <td>Physical Security Engineer</td>
      <td>Enercon Services, Inc.</td>
      <td>Birmingham, AL</td>
      <td>Birmingham, AL</td>
      <td>2025-11-21</td>
      <td>fulltime</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>li-4338291784</td>
      <td>linkedin</td>
      <td>https://www.linkedin.com/jobs/view/4338291784</td>
      <td>NaN</td>
      <td>Distinguished Engineer - AI Infrastructure Arc...</td>
      <td>Cisco</td>
      <td>Birmingham, AL</td>
      <td>Birmingham, AL</td>
      <td>2025-11-21</td>
      <td>fulltime</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>li-4302046287</td>
      <td>linkedin</td>
      <td>https://www.linkedin.com/jobs/view/4302046287</td>
      <td>NaN</td>
      <td>Information Services Quality Assurance Interns...</td>
      <td>Altec</td>
      <td>Birmingham, AL</td>
      <td>Birmingham, AL</td>
      <td>2025-11-21</td>
      <td>internship</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>



### Aggregating sampled data scientist jobs in comparison

This aggregation builds a dataset of sampled 836 data scientist jobs from SF, LA, Seattle, San Jose, New York and Denver to be compared to the aggregated backend engineering dataset.


```python
jobs_ds = pd.read_csv(root_dir+"jobs_ds.csv")
jobs_ds.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>site</th>
      <th>job_url</th>
      <th>job_url_direct</th>
      <th>title</th>
      <th>company</th>
      <th>location</th>
      <th>area</th>
      <th>date_posted</th>
      <th>job_type</th>
      <th>...</th>
      <th>company_addresses</th>
      <th>company_num_employees</th>
      <th>company_revenue</th>
      <th>company_description</th>
      <th>skills</th>
      <th>experience_range</th>
      <th>company_rating</th>
      <th>company_reviews_count</th>
      <th>vacancy_count</th>
      <th>work_from_home_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>li-4290341003</td>
      <td>linkedin</td>
      <td>https://www.linkedin.com/jobs/view/4290341003</td>
      <td>https://www.samsara.com/company/careers/roles/...</td>
      <td>(New Grad) Software Engineering</td>
      <td>Samsara</td>
      <td>San Francisco, CA</td>
      <td>Seattle</td>
      <td>2025-11-21</td>
      <td>fulltime</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>li-4338221974</td>
      <td>linkedin</td>
      <td>https://www.linkedin.com/jobs/view/4338221974</td>
      <td>NaN</td>
      <td>Software Engineer, Bridge</td>
      <td>Stripe</td>
      <td>San Francisco, CA</td>
      <td>Seattle</td>
      <td>2025-11-21</td>
      <td>fulltime</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>li-4333322957</td>
      <td>linkedin</td>
      <td>https://www.linkedin.com/jobs/view/4333322957</td>
      <td>NaN</td>
      <td>Full Stack Engineer</td>
      <td>Adobe</td>
      <td>San Jose, CA</td>
      <td>Seattle</td>
      <td>2025-11-21</td>
      <td>fulltime</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>li-4337602941</td>
      <td>linkedin</td>
      <td>https://www.linkedin.com/jobs/view/4337602941</td>
      <td>https://dsp.prng.co/1pYV21b&amp;urlHash=bseg</td>
      <td>Software Engineer 2</td>
      <td>Intuit</td>
      <td>Mountain View, CA</td>
      <td>Seattle</td>
      <td>2025-11-21</td>
      <td>fulltime</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>li-4338372388</td>
      <td>linkedin</td>
      <td>https://www.linkedin.com/jobs/view/4338372388</td>
      <td>https://dsp.prng.co/sqakw3b&amp;urlHash=7Bda</td>
      <td>Software Engineer I, Virtual Expert Platform (...</td>
      <td>Intuit</td>
      <td>Mountain View, CA</td>
      <td>Seattle</td>
      <td>2025-11-21</td>
      <td>fulltime</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>




```python
jobs_ds.shape
```




    (834, 35)




```python

```
