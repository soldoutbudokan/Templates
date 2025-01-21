# %%
################################################################################
# Title: Example of Web Scraping
# Author: Tirth Bhatt
# Date Created:
# Last Modified:
#
# Project: NBA Finals vs World Series - TV Ratings
# 
################################################################################

# %%
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

# %%
# Getting the NBA data
def get_nba_ratings():
    url = 'https://en.wikipedia.org/wiki/NBA_Finals_television_ratings'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    tables = soup.find_all('table', class_='wikitable')
    ratings_table = tables[1]
    
    data = []
    for row in ratings_table.find_all('tr')[1:]:
        cols = row.find_all(['td', 'th'])
        if len(cols) >= 4:
            try:
                year = int(cols[0].text.strip())
                network = cols[1].text.strip()
                teams = cols[2].text.strip().replace('\n', ' ').replace('  ', ' ')
                rating_text = cols[3].text.strip()
                
                viewers = None
                if 'Viewers' in rating_text:
                    viewers_part = rating_text.split('(')[-1].split(')')[0]
                    viewers = ''.join(filter(lambda x: x.isdigit() or x == '.', viewers_part))
                
                if year >= 1995:
                    data.append({
                        'Year': year,
                        'Network': network,
                        'Teams': teams,
                        'Viewers (M)': float(viewers) if viewers and viewers.replace('.', '').isdigit() else None
                    })
            except (ValueError, IndexError):
                continue
    
    df = pd.DataFrame(data)
    return df.sort_values('Year', ascending=False)

def main():
    df = get_nba_ratings()
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print("\nNBA Finals Viewers (1995-present):")
    print(df.to_string(index=False))
    
    return df

if __name__ == "__main__":
    nba_df = main()

# %%
# Getting the World Series data
def get_ws_ratings():
    url = 'https://en.wikipedia.org/wiki/World_Series_television_ratings'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    ratings_table = soup.find('table', class_='wikitable sortable')
    
    clean_ws = []
    for row in ratings_table.find_all('tr')[1:]:
        cols = row.find_all(['td', 'th'])
        if len(cols) >= 4:
            try:
                year = int(cols[0].text.strip())
                network = cols[1].text.strip()
                teams = cols[2].text.strip().replace('\n', ' ').replace('  ', ' ')
                rating_text = cols[3].text.strip()
                
                viewers = None
                if 'viewers' in rating_text.lower():
                    viewers_part = rating_text.split('(')[1].split(')')[0]
                    viewers = float(viewers_part.replace('M', '').replace('viewers', '').strip())
                
                clean_ws.append({
                    'Year': year,
                    'Network': network,
                    'Teams': teams,
                    'Viewers_M': viewers
                })
            except (ValueError, IndexError):
                continue
    
    return pd.DataFrame(clean_ws).sort_values('Year', ascending=False)

clean_ws = get_ws_ratings()
print("\nWorld Series Television Ratings:")
print(clean_ws.to_string(index=False))

# %%
# Merge the tables on Year
merged_df = nba_df.merge(
    clean_ws[['Year', 'Teams', 'Viewers_M']],
    on='Year',
    how='left',
    suffixes=('_NBA', '_WS')
)

# Rename columns for clarity
merged_df = merged_df.rename(columns={
    'Teams_NBA': 'NBA_Teams',
    'Teams_WS': 'WS_Teams',
    'Viewers (M)': 'NBA_Viewers',
    'Viewers_M': 'WS_Viewers'
})
    
print(merged_df)

# %%
# Visualizations
plt.figure(figsize=(15, 8))
plt.style.use('seaborn-v0_8-white')  # Clean base style

# Plot lines
plt.plot(merged_df['Year'], merged_df['NBA_Viewers'], 
         linewidth=2.5, color='#1f77b4', label='NBA Finals')
plt.plot(merged_df['Year'], merged_df['WS_Viewers'], 
         linewidth=2.5, color='#d62728', label='World Series')

# Remove gridlines
plt.grid(False)

# Bold title and clean labels
plt.title('NBA Finals vs World Series Viewership Trends', 
          fontsize=14, pad=20, fontweight='bold')
plt.xlabel('Year', fontsize=11)
plt.ylabel('Television Viewers (Millions)', fontsize=11)

# Legend with white background
plt.legend(fontsize=10, loc='upper right', 
          frameon=True, framealpha=1)

# Show all years on x-axis
all_years = range(merged_df['Year'].min(), merged_df['Year'].max() + 1)
plt.xticks(all_years, rotation=45, ha='right')

# Set y-axis limits
plt.ylim(5, 30)

plt.tight_layout()
plt.show()
# %%
