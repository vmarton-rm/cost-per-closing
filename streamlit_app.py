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

# Create the select boxes
selected_year = st.number_input("Select year", min_value=2020, max_value=2030, value=pd.Timestamp.today().year, step=1)
selected_month = st.selectbox("Select month", options=list(months.keys()))

# Convert selected month to numerical value
selected_month_num = months[selected_month]

purch_col, refi_col, renew_col = st.columns(3)
purch = purch_col.checkbox('Purchase')
refi = refi_col.checkbox('Refinance')
renew = renew_col.checkbox('Renewal')

sql_DT = 'select "YearMonth", "Lead Source"'
# if purch or refi or renew:
#     sql_DT = sql_DT + ', "Loan Purpose"'
sql_DT = sql_DT + ', sum("Leads") as "Leads", sum("Allocations") as "Allocations", sum("Credits") as "Credits"'
sql_DT = sql_DT + ', sum("Submissions") as "Submissions", sum("Closings") as "Closings", sum("Expected Closings") as "Expected Closings", sum("Cost") as "Cost"'
sql_DT = sql_DT + ' from production.analytical.lead_cost_breakdown_by_purpose'
if purch or refi or renew:
    sql_DT = sql_DT + ' where "Loan Purpose" in ('
    if purch:
        sql_DT = sql_DT + "'Purchase', "
    if refi:
        sql_DT = sql_DT + "'Refinance', "
    if renew:
        sql_DT = sql_DT + "'Renewal', "
    sql_DT = sql_DT[:-2] + ')'
sql_DT = sql_DT + ' group by "YearMonth", "Lead Source"'
# if purch or refi or renew:
#     sql_DT = sql_DT + ', "Loan Purpose"'
data_DT = session.sql(sql_DT).to_pandas()

# Assuming the date column is named 'YearMonth' and has the format 'YYYY MM'
data_DT['year'] = data_DT['YearMonth'].str[:4]
data_DT['month'] = data_DT['YearMonth'].str[5:7]

data_DT['Cost ($)'] = round(data_DT['Cost'])
data_DT['Cost per Lead ($)'] = round(data_DT['Cost'] / data_DT['Leads'])
data_DT['Cost per Allocation ($)'] = round(data_DT['Cost'] / data_DT['Allocations'])
data_DT['Cost per Credit ($)'] = round(data_DT['Cost'] / data_DT['Credits'])
data_DT['Cost per Submission ($)'] = round(data_DT['Cost'] / data_DT['Submissions'])
data_DT['Cost per Closing ($)'] = round(data_DT['Cost'] / data_DT['Closings'])
data_DT['Cost per Expected Closing ($)'] = round(data_DT['Cost'] / data_DT['Expected Closings'])
data_DT['% Lead to Allocate (cohort)'] = round(data_DT['Allocations'] / data_DT['Leads'] * 100, 2)
data_DT['% Lead to Credit (cohort)'] = round(data_DT['Credits'] / data_DT['Leads'] * 100, 2)
data_DT['% Lead to Submit (cohort)'] = round(data_DT['Submissions'] / data_DT['Leads'] * 100, 2)
data_DT['% Lead to Close (cohort)'] = round(data_DT['Closings'] / data_DT['Leads'] * 100, 2)
data_DT['% Expected Lead to Close (cohort)'] = round(data_DT['Expected Closings'] / data_DT['Leads'] * 100, 2)

data_DT.drop(columns=['Cost'], inplace=True)
data_DT = data_DT.fillna(0)
filtered_df = data_DT[(data_DT['year'] == str(selected_year)) & (data_DT['month'] == selected_month_num)].sort_values('Leads', ascending=False)

## Automate the Bake precentage

SQL_LTC_Bake = f'select * from production.analytical.LeadToClose_Bake'
SQL_LTC_Bake_DT = session.sql(SQL_LTC_Bake).to_pandas()

# Create the selected_date as the first day of the selected month
selected_date = pd.to_datetime(f"{selected_year}-{months[selected_month]}-01") + pd.offsets.MonthEnd(0)

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
  closest_row = df[df['TTDAYS'] <= days_since].max()

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

    # st.data_editor(
    #     df_to_display,
    #     column_config={
    #         'Cost ($)': st.column_config.NumberColumn(
    #             "Cost ($)",
    #             help='Total Lead Cost in CAD',
    #             min_value=0,
    #             step=1
    #         )
    #     },
    #     hide_index=True
    # )

    df_to_display = df_to_display.set_index(df_to_display.columns[0])
    
    
    st.dataframe(df_to_display.style.format(
        {
            'Cost ($)' : '${0:,.0f}',
            'Cost per Lead ($)' : '${0:,.0f}',
            'Cost per Allocation ($)' : '${0:,.0f}',
            'Cost per Credit ($)' : '${0:,.0f}',
            'Cost per Submission ($)' : '${0:,.0f}',
            'Cost per Closing ($)' : '${0:,.0f}',
            'Cost per Expected Closing ($)' : '${0:,.0f}',
            '% Lead to Allocate (cohort)' : '{0:.2f}%',
            '% Lead to Credit (cohort)' : '{0:.2f}%',
            '% Lead to Submit (cohort)' : '{0:.2f}%',
            '% Lead to Close (cohort)' : '{0:.2f}%',
            '% Expected Lead to Close (cohort)' : '{0:.2f}%'
        })
    )


    st.title("Cost Per Unit")
    units = {
        "Leads": ['Lead Source', 'Leads','Cost ($)', 'Cost per Lead ($)'],
        "Allocations": ['Lead Source', 'Leads','Allocations','Cost ($)', 'Cost per Allocation ($)', '% Lead to Allocate (cohort)'],
        "Credits": ['Lead Source', 'Leads','Credits','Cost ($)', 'Cost per Credit ($)', '% Lead to Credit (cohort)'],
        "Submissions": ['Lead Source', 'Leads','Submissions','Cost ($)', 'Cost per Submission ($)', '% Lead to Submit (cohort)'],
        "Closings": ['Lead Source', 'Leads','Closings','Cost ($)', 'Cost per Closing ($)', '% Lead to Close (cohort)'],
        "Expected Closings": ['Lead Source', 'Leads','Expected Closings','Cost ($)', 'Cost per Expected Closing ($)', '% Expected Lead to Close (cohort)']
    }
    selected_unit = st.selectbox("Select Unit", options=list(units.keys()))
    selected_columns = units[selected_unit]
    selected_to_display = filtered_df[selected_columns]
    selected_to_display = selected_to_display.set_index(selected_to_display.columns[0])

    format_dict = {
            'Cost ($)' : '${0:,.0f}',
            'Cost per ' + str(selected_unit)[:-1] + ' ($)' : '${0:,.0f}'
        }
    if selected_unit == 'Allocations':
        format_dict['% Lead to Allocate (cohort)'] = '{0:.2f}%'
    elif selected_unit == 'Credits':
        format_dict['% Lead to Credit (cohort)'] = '{0:.2f}%'
    elif selected_unit == 'Submissions':
        format_dict['% Lead to Submit (cohort)'] = '{0:.2f}%'
    elif selected_unit == 'Closings':
        format_dict['% Lead to Close (cohort)'] = '{0:.2f}%'
    elif selected_unit == 'Expected Closings':
        format_dict['% Expected Lead to Close (cohort)'] = '{0:.2f}%'
    
    st.dataframe(selected_to_display.style.format(format_dict))

lead_sources = session.sql('select distinct "Lead Source" from production.analytical.lead_cost_breakdown').to_pandas()
st.subheader('Cost Metrics over Time')

options = sorted(list(x.item() for x in lead_sources.values))
selected_source = st.selectbox('Select lead source', options=options, index=options.index('LowestRates'))
options_b = ['Cost per Lead ($)', 'Cost per Allocation ($)', 'Cost per Credit ($)',
             'Cost per Submission ($)', 'Cost per Closing ($)']
selected_metric = st.selectbox('Select metric', options=options_b, index=options_b.index('Cost per Closing ($)'))

st.markdown('The shaded regions represent the bake % of the cohort based on the selected metric, ' +
   'with full transparency signifying >95% bake.')
st.markdown('Click on any line in the legend to toggle its display.')

data_DT['Month'] = pd.to_datetime(data_DT['year'].str[:] + '-' + data_DT['month'].str[:]) + pd.tseries.offsets.MonthEnd(0)
graphed_df = data_DT[data_DT['Lead Source'] == str(selected_source)].sort_values('Month')
today = datetime.datetime.today().strftime('%Y-%m-%d')
graphed_df.sort_values(['Month'], inplace=True)

fig = go.Figure()

filled = graphed_df[['Month', str(selected_metric)]].sort_values(by='Month')
all_dates = [filled['Month'].min()]
most_recent = all_dates[0]
while most_recent.strftime('%Y-%m') != datetime.datetime.today().strftime('%Y-%m'):
    most_recent = most_recent + pd.Timedelta(days=1) + pd.tseries.offsets.MonthEnd(0)
    all_dates.append(most_recent)
filled = pd.DataFrame(all_dates, columns=['Month']).merge(filled, how='left', left_on='Month', right_on='Month').fillna(filled[str(selected_metric)].rolling(window=2, min_periods=1).mean())
filled.loc[filled[str(selected_metric)] == float('inf'), str(selected_metric)] = filled[str(selected_metric)].rolling(window=2, min_periods=1).mean()
filled.sort_values(by='Month')
graphed_df.loc[graphed_df[str(selected_metric)] == float('inf'), str(selected_metric)] = graphed_df[str(selected_metric)].rolling(window=2, min_periods=1).mean()
gaps_df = pd.DataFrame(all_dates, columns=['Month']).merge(graphed_df, how='left', left_on='Month', right_on='Month')

if(str(selected_metric) == 'Cost per Closing ($)'):
    fig.add_trace(
        go.Scatter(
            x=gaps_df['Month'],
            y=gaps_df['Cost per Expected Closing ($)'],
            fill=None, #fill="tonexty"
            mode="lines",
            line_color="pink",
            name='Cost per Expected Closing ($)',
            connectgaps=False
        )
    )

fig.add_trace(
    go.Scatter(
        x=gaps_df['Month'],
        y=gaps_df[str(selected_metric)],
        fill=None,
        mode="lines",
        line_color = 'lightblue',
        name=str(selected_metric),
        connectgaps=False
    )  
)

fig.add_trace(
    go.Scatter(
        x=graphed_df['Month'],
        y=graphed_df[str(selected_metric)].fillna(filled[str(selected_metric)].rolling(window=2, min_periods=1).mean()).rolling(window=13).mean(),
        fill=None,
        mode="lines",
        line_color="green",
        name='Moving Average (12 months)',
        visible='legendonly'
    )
)

fig.add_trace(
    go.Scatter(
        x=graphed_df['Month'],
        y=graphed_df[str(selected_metric)].fillna(filled[str(selected_metric)].rolling(window=2, min_periods=1).mean()).rolling(window=3).mean(),
        fill=None,
        mode="lines",
        line_color="blue",
        name='Moving Average (3 months)',
        visible='legendonly'
    )
)

last_day = datetime.datetime.today()
first_day = datetime.datetime.today().replace(day=1)
graph_days_since = (datetime.datetime.today() - first_day).days

bake_percentages = f'select * from production.analytical.'
if selected_metric == 'Cost per Allocation ($)':
    bake_percentages = bake_percentages + 'leadtoallocate_bake'
elif selected_metric == 'Cost per Credit ($)':
    bake_percentages = bake_percentages + 'leadtocredit_bake'
elif selected_metric == 'Cost per Submission ($)':
    bake_percentages = bake_percentages + 'leadtosubmit_bake'
elif selected_metric == 'Cost per Closing ($)':
    bake_percentages = bake_percentages + 'leadtoclose_bake'
else:
    bake_percentages = None

if bake_percentages is not None:
    bake_df = session.sql(bake_percentages).to_pandas()
else:
    bake_df = None

while bake_df is not None and float(get_cumulative_percent(graph_days_since, bake_df)) < 0.95:
    fig.add_shape(type='rect',
                  x0 = first_day.strftime('%Y-%m-%d'), x1=last_day.strftime('%Y-%m-%d'),
                  y0=0, y1=graphed_df[str(selected_metric)].max() * 1.05,
                  line=None, fillcolor="Red",
                  opacity=(1 - get_cumulative_percent(graph_days_since, bake_df)), layer='below')
    last_day = first_day + datetime.timedelta(days=-1)
    first_day = last_day.replace(day=1)
    graph_days_since = (datetime.datetime.today() - first_day).days

fig.update_xaxes(range=['2020-03-01', today])
fig.update_yaxes(range=[0, graphed_df[str(selected_metric)].max() * 1.05])
fig.update_layout(showlegend=True, legend=dict(
                                    title="Cost by Metric",
                                    traceorder="normal",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="left",
                                    x=0.02))
fig.show()
st.plotly_chart(figure_or_data=fig, use_container_width=True)

st.markdown('Note that the 12- and 3-month moving average plots do not begin until ' +
           'as many data points are available as the width of the moving window. This ' +
           'will cause the 12-month plot, for example, to begin more than one year ' +
           'after the first data point for lead sources which were temporarily ' +
           'disabled within one year of their activation. See: Borrowell')
