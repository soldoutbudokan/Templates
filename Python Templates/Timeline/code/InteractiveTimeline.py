################################################################################
# PRELIMINARY & UNPROOFED
# Not PRIVLEDGED & Not CONFIDENTIAL
# 
#   
# Case: Personal Interest/Training/Templates									
# Author: soldoutbudokan
# Version Number:
# Date Created: 						
# Last Updated: 						
#   
# Audited: No
# To run from cmd: python "C:\Users\[USERNAME]\Documents\Templates\Timeline\code\InteractiveTimeline.py"
################################################################################

# %%
# Importing Packages and setting WD
import plotly.graph_objects as go
import pandas as pd
import os
from datetime import timedelta

# Set the working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# %%
# Reading and cleaning files
# Read the events from the Excel file
df = pd.read_excel(os.path.join('input', 'timeline_events.xlsx'))

# Convert the 'Date' column to datetime and sort
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
# %%
# Setting Universal definitions
# Define a custom color map based on flag colors
color_map = {
    'Austria-Hungary': '#FACF00',
    'Germany': '#000000',
    'Russia': '#E41A1C',
    'Serbia': '#377EB8',
    'Britain': '#00247D'
}

# Make a list of unique countries
unique_countries = df['Country'].unique()
# %%
# Create the figure
fig = go.Figure()

# Add invisible traces for the legend
for country in unique_countries:
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=15, color=color_map[country]),
        name=country,
        legendgroup=country,
    ))

# Add events to the timeline
for i, row in df.iterrows():
    y_position = i * 1.5  # Slightly increase vertical spacing
    if row['Key Event'] != 'Yes':
        fig.add_trace(go.Scatter(
            x=[row['Date'], row['Date']],
            y=[y_position, y_position],
            mode='markers',
            marker=dict(size=15, color=color_map[row['Country']], symbol='circle'),
            hoverinfo='text',
            hovertext=f"{row['Event']}<br>{row['Date'].strftime('%m/%d/%Y')}<br>{row['Description']}",
            showlegend=False,
            legendgroup=row['Country'],
        ))
        
        # Add text with white background
        fig.add_annotation(
            x=row['Date'],
            y=y_position,
            text=row['Event'],
            showarrow=False,
            yshift=15,
            xanchor='center',
            yanchor='bottom',
            bgcolor='white',
            opacity=1,
            bordercolor='white',
            borderwidth=2,
            font=dict(color='black', size=11)
        )

    # Add vertical lines for key events
    if row['Key Event'] == 'Yes':
        fig.add_shape(
            type="line",
            x0=row['Date'],
            x1=row['Date'],
            y0=-1,
            y1=len(df) * 1.5 + 2,  # Adjust line height
            line=dict(color="black", width=3, dash="dash"),
        )
        fig.add_annotation(
            x=row['Date'],
            y=len(df) * 1.5 + 2,  # Adjust key event descriptor position
            text=f"<b>{row['Event']}</b>",
            showarrow=False,
            yshift=10,
            font=dict(size=14)
        )

# Calculate date range with extended start and end dates to fit all events
start_date = df['Date'].min() - timedelta(days=5)
end_date = df['Date'].max() + timedelta(days=5)
date_range = end_date - start_date
padding = timedelta(days=date_range.days * 0.05)

# Add box around the timeline
fig.add_shape(
    type="rect",
    xref="paper", yref="paper",
    x0=0, y0=0, x1=1, y1=1,
    line=dict(color="black", width=2),
    fillcolor="rgba(0,0,0,0)"
)

# Update layout
fig.update_layout(
    title=dict(
        text="<b>July Crisis and the Start of World War I</b>",
        font=dict(size=24),
        x=0.01,
        xanchor='left'
    ),
    xaxis_title="Date",
    yaxis=dict(
        visible=False,
        showticklabels=False
    ),
    hovermode="closest",
    height=900,
    margin=dict(l=50, r=50, t=120, b=50),
    xaxis=dict(
        type="date",
        range=[start_date - padding, end_date + padding],
        showgrid=True,
        gridcolor='#A9A9A9',
        gridwidth=2,
        tickformat='%m/%d/%Y'
    ),
    legend_title_text='Countries',
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Adjust y-axis range
fig.update_yaxes(range=[-1, len(df) * 1.5 + 2])
# %%
# Save the figure as an interactive HTML file
output_file = os.path.join('output', 'july_crisis_timeline.html')
fig.write_html(output_file, include_plotlyjs=True, full_html=True)

print(f"The interactive timeline has been saved as '{output_file}'")