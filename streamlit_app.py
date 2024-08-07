import datetime
import streamlit as st
import numpy as np
import pandas as pd
from snowflake.snowpark.context import get_active_session
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import time


# Write directly to the app
st.title("Cost Per Closing")

# Connect to Snowflake
conn = st.connection("snowflake")


# @st.cache_data
def load_session():
    session = conn.session()
    return session


session = load_session()

# Use an interactive slider to get user input

months = {
    "January": "01",
    "February": "02",
    "March": "03",
    "April": "04",
    "May": "05",
    "June": "06",
    "July": "07",
    "August": "08",
    "September": "09",
    "October": "10",
    "November": "11",
    "December": "12"
}

sql_DT = f'select * from PRODUCTION.ANALYTICAL.LEAD_COST_BREAKDOWN'

data_DT = session.sql(sql_DT).to_pandas().sort_values('Leads', ascending=False)

# Create the select boxes
selected_year = st.number_input("Select year", min_value=2020, max_value=2030, value=2024, step=1)
selected_month = st.selectbox("Select month", options=list(months.keys()))

# Convert selected month to numerical value
selected_month_num = months[selected_month]

# Assuming the date column is named 'YearMonth' and has the format 'YYYY MM'
data_DT['year'] = data_DT['YearMonth'].str[:4]
data_DT['month'] = data_DT['YearMonth'].str[5:7]
filtered_df = data_DT[(data_DT['year'] == str(selected_year)) & (data_DT['month'] == selected_month_num)]



## Automate the Bake precentage

SQL_LTC_Bake = f'select * from PRODUCTION.ANALYTICAL.LeadToClose_Bake'
SQL_LTC_Bake_DT = session.sql(SQL_LTC_Bake).to_pandas()

# Create the selected_date as the first day of the selected month
selected_date = pd.to_datetime(f"{selected_year}-{months[selected_month]}-01")
# Calculate the difference between today and the selected date
days_since = (pd.Timestamp.today() - selected_date).days


def get_cumulative_percent(days_since, df):
  """
  Gets the cumulative percent based on days since from the provided DataFrame.

  Args:
    days_since: The number of days since.
    df: The DataFrame containing the lookup data.

  Returns:
    The cumulative percent value.
  """

  # Find the closest value in the DataFrame
  closest_row = df[df['LEADTOCLOSETTDAYS'] <= days_since].max()

  # Return the cumulative percent for the closest row
  return closest_row['CUMULATIVE_PERCENT']


with st.spinner('Please wait...'):
    cumulative_percent = get_cumulative_percent(days_since, SQL_LTC_Bake_DT)
    st.title("Bake %")
    st.metric(label = 'Bake % is a measure to track how close we are to finishing a cohort month.',
              label_visibility='visible', value=str(round(cumulative_percent * 100, 2)) + '%')

    layout = go.Layout(
        xaxis = go.XAxis(
            title = 'Bake %'),
        yaxis = go.YAxis(
            showticklabels=False
        )
    )
    progress_figure = go.Figure(layout=layout)
    progress_figure.add_shape(type='rect', 
                             x0=0, x1=cumulative_percent*100, y0=0, y1=1,
                             line=None, fillcolor='LawnGreen')
    progress_figure.add_shape(type='rect', 
                             x0=cumulative_percent*100, x1=100, y0=0, y1=1,
                             line=None, fillcolor='Red')
    progress_figure.update_xaxes(range=[0,100])
    progress_figure.update_yaxes(range=[0,1])
    progress_figure.update_layout(height=50, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(progress_figure)    
    time.sleep(1)
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_to_display = filtered_df.copy()
    
    # Remove the 'Customer' column
    df_to_display = df_to_display.drop(['YearMonth','year','month'], axis=1)
    
    st.dataframe(df_to_display.set_index(df_to_display.columns[0]))

lead_sources = session.sql('select distinct "Lead Source" from production.analytical.lead_cost_breakdown').to_pandas()
st.subheader('Cost per Closing over Time')
options = sorted(list(x.item() for x in lead_sources.values))
selected_source = st.selectbox('Select lead source', options=options, index=options.index('LowestRates'))

data_DT['First Day of Month'] = data_DT['year'].str[:] + '-' + data_DT['month'].str[:] + '-01'
graphed_df = data_DT[data_DT['Lead Source'] == str(selected_source)]#['First Day of Month', 'Cost per Closing']
today = datetime.datetime.today().strftime('%Y-%m-%d')

fig = px.line(graphed_df, x='First Day of Month', y='Cost per Closing')

last_day = datetime.datetime.today()
first_day = datetime.datetime.today().replace(day=1)
graph_days_since = (datetime.datetime.today() - first_day).days

while float(get_cumulative_percent(graph_days_since, SQL_LTC_Bake_DT)) < 0.95:
    fig.add_shape(type='rect',
                  x0 = first_day.strftime('%Y-%m-%d'), x1=last_day.strftime('%Y-%m-%d'),
                  y0=0, y1=graphed_df['Cost per Closing'].max() * 1.05,
                  line=None, fillcolor="OrangeRed",
                  opacity=(1 - get_cumulative_percent(graph_days_since, SQL_LTC_Bake_DT)), layer='below')
    last_day = first_day + datetime.timedelta(days=-1)
    first_day = last_day.replace(day=1)
    graph_days_since = (datetime.datetime.today() - first_day).days

fig.update_xaxes(range=['2020-03-01', today])
fig.update_yaxes(range=[0, graphed_df['Cost per Closing'].max() * 1.05])
fig.update_layout(showlegend=True, legend=dict(title="Legend Title", traceorder="normal"))
fig.show()
st.plotly_chart(figure_or_data=fig, use_container_width=True)
st.markdown('The shaded regions represent the bake % of the cohort, with full transparency signifying '+
       '>95% bake.')
