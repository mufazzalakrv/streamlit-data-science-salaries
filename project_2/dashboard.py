import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sea
from matplotlib.widgets import Slider
import altair as alt
import pycountry
import warnings
warnings.filterwarnings("ignore")


#"""Data Loading and Data Cleaning"""
DATA = Path(__file__).resolve().parent.parent / "data2" / "DataScience_salaries_2025.csv"
salaries = pd.read_csv(DATA)


#"""Streamlit Dashboard Configuration"""
st.set_page_config(
    page_title="Dashboard - Project",
    layout="wide", 
    page_icon=":bar_chart:",
    initial_sidebar_state="expanded"
)

st.title(" ")
st.title("Data Science Salaries Exploratory Analysis 2023-2025")

#"""Creating the sidebar filters for the dashboard"""
st.sidebar.header("Filters")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview","EDA", "Remoteness vs. Years",
                                "Salaries Breakdown", "Comparison Analysis"])



#--- 1. Cleaning the data

## Check for duplicated rows in the dataset
num_duplicated = salaries.duplicated(keep = 'first').sum()
if num_duplicated > 1:
    duplicated_rows = salaries[salaries.duplicated(keep = 'first')]
    print(f'\nNumber of duplicated rows: {num_duplicated}')
else:
    print("No duplicated rows found.")

## Drop duplicated rows
if num_duplicated > 1:
    salaries = salaries.drop_duplicates(keep = 'first')
    print(f'Duplicated rows dropped. New number of rows: {len(salaries)}') 
# ---salaries.to_csv(DATA, index=False)

## Check the number of missing values
num_missing = salaries.isnull().sum().sum()
print(f'\nNumber of missing values: {num_missing}')

#drop missing values if any
#if num_missing > 0:
#    salaries = salaries.dropna()
#    print(f'Missing values dropped. New number of rows: {len(salaries)}')
# Output: Number of missing values: 0

#drop years 2020, 2021, 2022, because data is heavily unbalanced for these years
salaries = salaries[~salaries.work_year.isin([2020, 2021, 2022])]
# ---salaries.to_csv(DATA, index=False)


#--- 2. Data transformation and categorization
salaries["remote_label"] = salaries["remote_ratio"].map(
    {0: "On-site", 50: "Hybrid", 100: "Fully remote"}
)
order = ["On-site", "Hybrid", "Fully remote"]
# ---salaries.to_csv(DATA, index=False)

from countries import COMPANY_LOCATIONS
salaries["company_location"] = salaries["company_location"].map(COMPANY_LOCATIONS).fillna(salaries["company_location"])
# ---salaries.to_csv(DATA, index=False)

from countries import RESIDENCE_NAMES
salaries["employee_residence"] = salaries["employee_residence"].map(RESIDENCE_NAMES).fillna(salaries["employee_residence"])
# ---salaries.to_csv(DATA, index=False)



#"""Exploratory Data Analysis (EDA)"""

####  Visualizations and Insights

with tab1:
    st.markdown("<h1 style='color: #006400; text-align: left; font-weight: 700;'> Project Overview </h1>", 
        unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .overview-intro {
            background-color: #f4fbf6;
            color: #111111;
            border: 1.5px solid #95d5b2;
            border-left: 6px solid #2d6a4f;
            border-radius: 14px;
            padding: 20px 24px;
            box-shadow: 0 3px 8px rgba(0, 0, 0, 0.06);
            margin-bottom: 16px;
        }
        .overview-intro h3 {
            color: #006400;
            margin: 0 0 10px 0;
            font-weight: 700;
        }
        .overview-intro p {
            color: #111111;
            margin: 0;
            line-height: 1.55;
            font-size: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="overview-intro">
            <h3>Data Science Salaries: Global Market Analysis</h3>
            <p>
            This dataset provides insights into global data science
            job salaries from <b>2023 to 2025</b>. It includes information
            about experience levels, employment types, job titles,
            salary currencies, remote work arrangements and company
            characteristics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    
    st.markdown("<h1 style='color: #006400; text-align: left; font-weight: 700;'> Dataset Variables </h1>", 
        unsafe_allow_html=True)
    with st.expander("Explore all dataset variables", expanded=True):
        variables = pd.DataFrame({
            "Variable": [
                "work_year",
                "experience_level",
                "employment_type",
                "job_title",
                "salary",
                "salary_currency",
                "salary_in_usd",
                "employee_residence",
                "remote_ratio",
                "company_location",
                "company_size"
            ],
            "Description": [
                "The year in which the salary data was recorded.",
                "The employee's experience level (Entry, Mid, Senior, Executive).",
                "The type of employment (Full-time, Part-time, Contract, etc.).",
                "The employee's role within the data science field.",
                "The original salary amount.",
                "The currency in which the salary is paid.",
                "The salary converted into US dollars for comparison.",
                "The country where the employee resides.",
                "The percentage of remote work allowed for the position.",
                "The country where the company is located.",
                "The size of the company based on employee count."
            ],
            "Variable Type": [
                "Numeric / Time",
                "Ordinal",
                "Nominal",
                "Nominal",
                "Numeric",
                "Nominal",
                "Numeric",
                "Nominal",
                "Numeric",
                "Nominal",
                "Ordinal"
            ]
        })
        st.dataframe(variables,
            column_config = {"Variable" : st.column_config.MultiselectColumn("Variable Name", color = "blue")},
            use_container_width=True, hide_index=True)

    st.markdown("<h1 style='color: #006400; text-align: left; font-weight: 700;'> Aim of the Analysis </h1>", 
        unsafe_allow_html=True)

    st.markdown("""<style>
        .aim-flashcard {
            background-color: #d8f3dc;
            color: #000000;
            border: 1.5px solid #74c69d;
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 3px 8px rgba(0, 0, 0, 0.06);
            height: 100%;}
        .aim-flashcard-tag {
            display: inline-block;
            background-color: #b7e4c7;
            color: #000000;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 999px;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.04em;}
        .aim-flashcard-title {
            color: #000000;
            font-size: 1.1rem;
            font-weight: 700;
            margin: 0 0 8px 0;}
        .aim-flashcard-body {
            color: #000000;
            font-size: 0.96rem;
            line-height: 1.45;
            margin: 0;} </style> """, unsafe_allow_html=True,)

    flashcards = [("Focus 01", "Data Science Market",
            "Comprehensive data analysis and insights across 2023-2025",),
        ("Focus 02", "Job Tendencies", "Evaluation of job trends and salaries across countries",),
        ("Focus 03", "Comparative Analysis", "How various aspects differ and real-life logic application",),]

    card_cols = st.columns(3)
    for col, (tag, title, body) in zip(card_cols, flashcards):
        with col:
            st.markdown(f"""<div class="aim-flashcard"> <span class="aim-flashcard-tag">{tag}</span>
                <h4 class="aim-flashcard-title">{title}</h4> <p class="aim-flashcard-body">{body}</p>
                </div> """, unsafe_allow_html=True,)


    st.markdown("""
    <style>
    /* the metric card */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #6A5ACD;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
    }
    /* center the label container and text */
    [data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    /* the label */
    [data-testid="stMetricLabel"] p {
        color: #6A5ACD;
        font-weight: 800;
        font-size: 1.1rem;
    }
    /* the big value */
    [data-testid="stMetricValue"] {
        color: #2F4F4F;
        font-size: 1.6rem;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] p, [data-testid="stMetricValue"] {
        text-align: center;
        }
    /* center delta text */
    [data-testid="stMetricDelta"] {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)


with tab2:
    #creating a button with years filter
    # how to create with year filter, a filter for company size that they work simultaneously
    # add filters that work together: company location, employemnt type, experience level
    st.markdown("<h1 style='color: #006400; text-align: left; font-weight: 700;'> Filters </h1>", 
        unsafe_allow_html=True)
    with st.container():
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            selected_company_size = st.selectbox("Company Size", salaries['company_size'].unique())
            company_size_pick = salaries[salaries['company_size'] == selected_company_size]
        with col2:
            selected_year = st.selectbox("Years", company_size_pick['work_year'].unique())
            year_pick = company_size_pick[company_size_pick['work_year'] == selected_year]
        with col3:
            selected_location = st.multiselect("Company Location", year_pick['company_location'].unique(),
                                        default = year_pick['company_location'].unique())
            location_pick = year_pick[year_pick['company_location'].isin(selected_location)]
        with col4:
            selected_employment_type = st.multiselect("Employment Type", location_pick['employment_type'].unique(),
                                            default = location_pick['employment_type'].unique())
            employment_pick = location_pick[location_pick['employment_type'].isin(selected_employment_type)]
        with col5:
            selected_experience_level = st.multiselect("Experience Level", employment_pick['experience_level'].unique(),
                                            default = employment_pick['experience_level'].unique())
            year_pick = employment_pick[employment_pick['experience_level'].isin(selected_experience_level)]

    mean_salary = round(year_pick["salary_in_usd"].mean())
    median_salary = round(year_pick["salary_in_usd"].median())
    range_salary = round(year_pick["salary_in_usd"].max() - year_pick["salary_in_usd"].min())
    min_salary = round(year_pick["salary_in_usd"].min())
    min_salary_location = year_pick["company_location"].iloc[year_pick["salary_in_usd"].argmin()]
    max_salary = round(year_pick["salary_in_usd"].max())
    max_salary_location = year_pick["company_location"].iloc[year_pick["salary_in_usd"].argmax()]
    popular_job = year_pick["job_title"].value_counts().index[0]
    num_unique_jobs = year_pick["job_title"].nunique()
    n_unique_residence = year_pick["employee_residence"].nunique()
    n_unique_comp_location = year_pick["company_location"].nunique()
    freq_exp_lvl = year_pick["experience_level"].value_counts().index[0]
    freq_residence = year_pick["employee_residence"].value_counts().index[0]
    freq_comp_location = year_pick["company_location"].value_counts().index[0]
    
    st.markdown(
        "<h1 style='color: #006400; text-align: left; font-weight: 700;'> Preview Analysis of Job Market for Data Scientists </h1>", 
        unsafe_allow_html=True)
    
    # EDA METRICS
    with st.markdown('# Metrics'):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Most Popular Job Yearly", popular_job, border = True)
        col2.metric("# Job Titles", num_unique_jobs, border = True)
        col3.metric("Frequent Experience Level", freq_exp_lvl, border = True)
        col4.metric("Salary Range", f'${range_salary:,}', border = True)
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Min Salary", f'${min_salary:,} ({min_salary_location})', border = True)
        col2.metric("Mean Salary", f'${mean_salary:,}', border = True)
        col3.metric("Median Salary", f'${median_salary:,}', border = True)
        col4.metric("Max Salary", f'${max_salary:,} ({max_salary_location})', border = True)
    
    st.markdown(
        "<h1 style='color: #006400; text-align: left; font-weight: 650;'> In Which Currencies Employees were Paid the Most? </h1>", 
        unsafe_allow_html=True)
    
    # BOX PLOT: currencies 
    st.header("Box Plot of Salary Distribution by Currency")
    currency_counts = year_pick["salary_currency"].value_counts()
    top_currencies = currency_counts[currency_counts >= 5].index
    currency_data = year_pick[year_pick["salary_currency"].isin(top_currencies)]
    fig1 = px.box(
        currency_data,
        x="salary_currency",
        y="salary_in_usd",
        color="salary_currency",
        labels={
            "salary_currency": "Salary Currency",
            "salary_in_usd": "Salary (USD)"
        }, color_discrete_sequence=px.colors.qualitative.Bold)
    fig1.update_layout(height=500, showlegend=False, xaxis_title="Currency", yaxis_title="Salary (USD)")
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown(
        "<h1 style='color: #006400; text-align: left; font-weight: 650;'> How do Top Job Titles Vary from Salary Median? </h1>", 
        unsafe_allow_html=True)
    
    # BAR CHART: How top job titles vary from salary median
    st.header("Bar Chart of Top 20 Job titles and Median Salary")
    overall_median = salaries["salary_in_usd"].median()
    top_titles = salaries["job_title"].value_counts().head(20).index
    title_gap = (
        salaries[salaries["job_title"].isin(top_titles)]
            .groupby("job_title")["salary_in_usd"].median()
            .reset_index(name="median_salary"))
    title_gap["gap_pct"] = (title_gap["median_salary"] / overall_median - 1) * 100
    title_gap["direction"] = np.where(title_gap["gap_pct"] >= 0, "Above overall median", "Below overall median")
    title_gap = title_gap.sort_values("gap_pct")
    fig_gap = px.bar(
        title_gap,
        x="gap_pct",
        y="job_title",
        color="direction",
        orientation="h",
        color_discrete_map={"Above overall median": "#2a9d8f", "Below overall median": "#e76f51"},
        labels={"gap_pct": "Difference from overall median (%)", "job_title": "Job Title", "direction": ""})
    st.plotly_chart(fig_gap, width="stretch")



with tab3:
    st.markdown("<h1 style='color: #006400; text-align: left; font-weight: 700;'> Filters </h1>", 
        unsafe_allow_html=True)
    with st.container():
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            company_size_t3 = st.selectbox("Company Size  ", salaries["company_size"].unique())
            company_pick_t3 = salaries[salaries["company_size"] == company_size_t3]
        with col2:
            available_years = sorted(company_pick_t3["work_year"].unique())
            select_year_t3 = st.multiselect("Years  ", available_years, default=available_years)
            year_pick_t3 = company_pick_t3[company_pick_t3["work_year"].isin(select_year_t3)]
        with col3:
            select_location_t3 = st.multiselect("Company Location  ", year_pick_t3["company_location"].unique(),
                                        default=year_pick_t3["company_location"].unique())
            location_pick_t3 = year_pick_t3[year_pick_t3["company_location"].isin(select_location_t3)]
        with col4:
            select_employment_t3 = st.multiselect("Employment Type  ", location_pick_t3["employment_type"].unique(),
                                        default=location_pick_t3["employment_type"].unique())
            employment_pick_t3 = location_pick_t3[location_pick_t3["employment_type"].isin(select_employment_t3)]
        with col5:
            experience_lvl_t3 = st.multiselect("Experience Level    ", employment_pick_t3["experience_level"].unique(),
                                        default=employment_pick_t3["experience_level"].unique())
            filtered_t3 = employment_pick_t3[employment_pick_t3["experience_level"].isin(experience_lvl_t3)]

    all_years_t3 = company_pick_t3[company_pick_t3['company_location'].isin(select_location_t3)
                        & company_pick_t3['employment_type'].isin(select_employment_t3)
                        & company_pick_t3['experience_level'].isin(experience_lvl_t3)]
    
    st.markdown(
        "<h1 style='color: #006400; text-align: left; font-weight: 700;'> What is the Tendency of Remote Work Ratio? </h1>", 
        unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Heatmap of Remote Work Ratio")
            #HEATMAP
            count_pivot = filtered_t3.pivot_table(
                index='remote_ratio', columns='work_year', values='salary_in_usd', aggfunc='count', fill_value=0).reindex([100, 50, 0], fill_value=0)
            pct_pivot = count_pivot.div(count_pivot.sum(axis=0), axis=1).fillna(0) * 100
            cell_text = [
                [f"<b>{pct_pivot.loc[r, c]:.1f}%</b><br>({count_pivot.loc[r, c]:,})" for c in count_pivot.columns]
                for r in count_pivot.index]
            fig_hm = px.imshow(
                pct_pivot,
                x=[str(y) for y in pct_pivot.columns],
                y=["Remote", "Hybrid", "On-site"],
                color_continuous_scale="YlGnBu",
                aspect="auto",
                labels={"x": "Years", "y": "Work Type", "color": "Share (%)"})
            fig_hm.update_traces(
                text=cell_text,
                texttemplate="%{text}",
                hovertemplate="Year: %{x}<br>Type: %{y}<br>Share: %{z:.1f}%<extra></extra>")
            fig_hm.update_layout(margin=dict(t=30, b=20, l=20, r=20))
            fig_hm.update_xaxes(dtick=1, tickformat="d")
            st.plotly_chart(fig_hm, use_container_width=True)

        with col2:
            # SIMPLE LINE CHART
            st.subheader("Remote Work Trends")
            remote_trend = (all_years_t3.groupby("work_year")["remote_ratio"].mean().reset_index())
            fig = px.line(
                remote_trend,
                x="work_year",
                y="remote_ratio",
                markers=True,
                labels={
                    "work_year": "Year",
                    "remote_ratio": "Average Remote Ratio (%)"})
            fig.update_layout(yaxis_range=[0, 100], hovermode="x unified")
            fig.update_xaxes(dtick=1, tickformat="d")
            st.plotly_chart(fig, width = "content")
        
        with col3:
            # STACKED BAR CHART
            st.subheader("Distribution of Work Types Over Years")
            remote_distribution = (all_years_t3.groupby(["work_year", "remote_label"]).size().reset_index(name="Employees"))
            fig2 = px.bar(
                remote_distribution,
                x="work_year",
                y="Employees",
                color="remote_label",
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={
                    "work_year": "Year",
                    "remote_label": "Work Type",
                    "Employees": "Percentage of Employees"})
            fig2.update_layout(barmode="relative", barnorm="percent")
            st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown(
        "<h1 style='color: #006400; text-align: left; font-weight: 650;'> How is Employment Type Different from each Country? </h1>", 
        unsafe_allow_html=True)
    
        # BUBBLE CHART
    if all_years_t3.empty:
        st.warning("No data matches these filters.")
    else:
        st.header("Bubble Chart of Employment Types and Jobs")
        top_n_jobs = 10     
        employment_names = {"FT": "Full-time", "PT": "Part-time", "CT": "Contract", "FL": "Freelance"}
        countries_t5 = all_years_t3["company_location"].value_counts().index    # most common first
        def job_bubble_chart(country):
            country_df = all_years_t3[all_years_t3["company_location"] == country]
            top_jobs = country_df["job_title"].value_counts().head(top_n_jobs).index.tolist()
            bubble_data = (
                country_df[country_df["job_title"].isin(top_jobs)]
                .groupby(["work_year", "job_title", "employment_type"], as_index=False)
                .agg(median_salary=("salary_in_usd", "median"), records=("salary_in_usd", "size")))
            bubble_data["employment_type"] = bubble_data["employment_type"].map(employment_names)
            fig = px.scatter(
                bubble_data,
                x= "work_year",
                y="job_title",
                size="median_salary",
                color="employment_type",
                size_max=30,
                opacity=0.75,
                category_orders={"job_title": top_jobs},            
                hover_data={"median_salary": ":,.0f", "records": True},
                labels={"work_year": "Work Year",
                        "job_title": "Job Title",
                        "employment_type": "Employment Type",
                        "median_salary": "Median Salary (USD)",
                        "records": "Records"},
                color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_xaxes(dtick=1, tickformat="d", title="Work Year")
            fig.update_yaxes(title="")
            fig.update_layout(height=500, template="plotly_white", legend_title="Employment Type")
            return fig
        col1, col2 = st.columns(2)
        with col1:
            country_1 = st.selectbox("Country 1", countries_t5, index=0, key="t5_jobs_country_1")
            st.plotly_chart(job_bubble_chart(country_1), width="stretch", key="t5_jobs_chart_1")
        with col2:
            country_2 = st.selectbox("Country 2", countries_t5, index=min(1, len(countries_t5) - 1), key="t5_jobs_country_2")
            st.plotly_chart(job_bubble_chart(country_2), width="stretch", key="t5_jobs_chart_2")





with tab4:
    st.markdown("<h1 style='color: #006400; text-align: left; font-weight: 700;'> Filters </h1>", 
        unsafe_allow_html=True)
    with st.container():
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            company_size_t4 = st.selectbox("Company Size     ", salaries['company_size'].unique())
            company_pick_t4 = salaries[salaries['company_size'] == company_size_t4]
        with col2:
            select_year_t4 = st.selectbox("Years     ", company_pick_t4['work_year'].unique())
            year_pick_t4 = company_pick_t4[company_pick_t4['work_year'] == select_year_t4]
        with col3:
            select_location_t4 = st.multiselect("Company Locations     ", year_pick_t4['company_location'].unique(), 
                                            default = year_pick_t4['company_location'].unique())
            location_pick_t4 = year_pick_t4[year_pick_t4['company_location'].isin(select_location_t4)]
        with col4:
            select_employment_t4 = st.multiselect("Employment Types     ", location_pick_t4['employment_type'].unique(), 
                                            default = location_pick_t4['employment_type'].unique())
            employment_pick_t4 = location_pick_t4[location_pick_t4['employment_type'].isin(select_employment_t4)]
        with col5:
            experience_lvl_t4 = st.multiselect("Experience Level     ", employment_pick_t4['experience_level'].unique(), 
                                            default = employment_pick_t4['experience_level'].unique())
            year_pick_t4 = employment_pick_t4[employment_pick_t4['experience_level'].isin(experience_lvl_t4)]
    
    all_years_t4 = company_pick_t4[
        company_pick_t4['company_location'].isin(select_location_t4)
        & company_pick_t4['employment_type'].isin(select_employment_t4)
        & company_pick_t4['experience_level'].isin(experience_lvl_t4)]
    
    # filter variables for top countries and top jobs
    top_countries = (year_pick_t4["employee_residence"].value_counts().head(10).index)
    top_jobs = (year_pick_t4["job_title"].value_counts().head(8).index)
    residence_data = year_pick_t4[
            (year_pick_t4["employee_residence"].isin(top_countries)) &
            (year_pick_t4["job_title"].isin(top_jobs))]
    residence_jobs = (residence_data.groupby(["employee_residence", "job_title"]).size().reset_index(name="Employee_Count"))

    st.markdown(
            "<h1 style='color: #006400; text-align: left; font-weight: 700;'> How do Data Science Jobs vary Across Countries? </h1>", 
            unsafe_allow_html=True)
    # DATAFRAME FOR DISPLAY
    display_df = residence_jobs.sort_values("Employee_Count", ascending=False).copy()
    display_df["job_title"] = display_df["job_title"].apply(lambda title: [title])
    def color_count(value):
        if value > 50:
            return "color: green"
        if value < 50:
            return "color: red"
        return ""  
    styled_df = display_df.style.map(color_count, subset=["Employee_Count"])
    with st.expander("Most Popular Jobs by Country"):
            st.dataframe(
                styled_df,
                column_config = {"employee_residence": st.column_config.TextColumn("Employee Residence"),
                                "job_title": st.column_config.MultiselectColumn("Job Title", options=set(display_df["job_title"].explode().to_list()), color="auto"),
                                "Employee_Count": st.column_config.NumberColumn("Employee Count", format = "%d"),},
                hide_index = True,
                use_container_width=True)
    st.header("Treemap of Job Positions by Country")
    fig = px.treemap( 
        residence_jobs,
        path=["employee_residence", "job_title"],
        values="Employee_Count",
        color="job_title",
        color_discrete_sequence=px.colors.qualitative.Bold,
        hover_data={"Employee_Count": True})
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)


    if year_pick_t4.empty:
        st.warning("No data matches these filters.")
    else:
        st.markdown("<h1 style='color: #006400; text-align: left; font-weight: 700;'> 3D Map Overview by Median Salary and Job Positions </h1>", 
                unsafe_allow_html=True)
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.header("Median Salaries Across World")
                map_df = (
                        year_pick_t4.groupby("company_location", as_index=False)
                                .agg(median_salary=("salary_in_usd", "median"), records=("salary_in_usd", "size")))
                fig = px.choropleth(
                        map_df,
                        locations="company_location",      
                        locationmode="country names",
                        color="median_salary",
                        hover_name="company_location",
                        hover_data={"company_location": False, "records": True},
                        color_continuous_scale="Turbo",
                        projection="natural earth",
                        labels={"median_salary": "Median Salary"})
                fig.update_geos(showland=True,
                        landcolor="rgb(230, 230, 230)",
                        showocean=True,
                        oceancolor="rgb(240, 245, 250)",
                        showcountries=True,
                        countrycolor="white",
                        showcoastlines=True,
                        coastlinecolor="gray")
                st.plotly_chart(fig, width="stretch", height = 600)
            with col2:
                st.header("Jobs Popularity by Country")
                min_employees = 5
                n_titles = 20
                country_jobs = (year_pick_t4.groupby(["company_location", "job_title"]).size().reset_index(name="Employee_Count"))
                totals = year_pick_t4["company_location"].value_counts()
                top_jobs = (country_jobs.sort_values("Employee_Count", ascending=False)
                        .drop_duplicates("company_location").copy())
                top_jobs["Country_Total"] = top_jobs["company_location"].map(totals)
                top_jobs = top_jobs[top_jobs["Country_Total"] >= min_employees]
                common = top_jobs["job_title"].value_counts().head(n_titles).index
                top_jobs["job_group"] = top_jobs["job_title"].where(top_jobs["job_title"].isin(common), "Other")
                top_jobs["marker_size"] = np.sqrt(top_jobs["Employee_Count"])
                fig4 = px.scatter_geo(
                        top_jobs,
                        locations="company_location",
                        locationmode="country names",
                        color="job_group",
                        size="marker_size",
                        hover_name="company_location",
                        hover_data={
                            "job_group": True,
                            "Employee_Count": True,
                            "Country_Total": True,
                            "marker_size": False,
                            "job_group": False},
                        projection="orthographic",
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        size_max=35,
                        labels={"job_group": "Most Popular Job", "job_title" : "Job Title", "Employee_Count": "Number of Employees", "Country_Total": "Employees in Country"})
                fig4.update_geos(
                        showland=True,
                        landcolor="rgb(230, 230, 230)",
                        showocean=True,
                        oceancolor="rgb(240, 245, 250)",
                        showcountries=True,
                        countrycolor="white",
                        showcoastlines=True,
                        coastlinecolor="gray")
                fig4.update_layout(
                        legend_title="Most Popular Job",
                        legend=dict(font=dict(size=20), itemsizing="constant"))
                st.plotly_chart(fig4, width="stretch", height = 700)

        st.markdown("<h1 style='color: #006400; text-align: left; font-weight: 700;'> How do Company Location and Experience Level Affect Salaries? </h1>", 
                    unsafe_allow_html=True)     
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Top 5 Countries Median Salaries")
            line_countries = all_years_t4["company_location"].value_counts().head(5).index
            country_df = all_years_t4[all_years_t4["company_location"].isin(line_countries)]
            median_salary_by_year = (
                country_df.groupby(["work_year", "company_location"], as_index=False)
                .agg(median_salary=("salary_in_usd", "median")))
            fig = px.line(
                median_salary_by_year,
                x="work_year",
                y="median_salary",
                color="company_location",
                labels = {"company_location" : "Company Location"},
                markers=True)
            fig.update_xaxes(dtick=1, tickformat="d")
            fig.update_layout(yaxis_range = [40000, 180000], xaxis_title = "Work Year", yaxis_title = "Median Salary")
            st.plotly_chart(fig, use_container_width=True)
        # PIE CHART
        with col2:
            st.subheader("World Currency Distribution")
            min_share = 1.0
            currency_counts = year_pick_t4["salary_currency"].value_counts().reset_index()
            currency_counts.columns = ["salary_currency", "count"]
            share = currency_counts["count"] / np.sum(currency_counts["count"]) * 100
            currency_counts["salary_currency"] = currency_counts["salary_currency"].where(share >= min_share, "Other")
            currency_counts = currency_counts.groupby("salary_currency", as_index=False)["count"].sum()
            currency_counts["is_other"] = currency_counts["salary_currency"] == "Other"
            fig = px.pie(
                currency_counts,
                names="salary_currency",
                values="count",
                hole = 0.4)
            fig.update_traces(sort=False, textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        # linear graph with experience level and median salary
        with col3:
            st.subheader("Median Salary by Experience Level")
            experience_markers = {"EN": "circle", "MI": "square", "SE": "diamond", "EX": "triangle-up"}
            experience_labels = {"EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"}
            median_salary_by_experience = (
                all_years_t4.groupby(["work_year", "experience_level"], as_index=False)
                    .agg(median_salary=("salary_in_usd", "median")))
            years = sorted(all_years_t4["work_year"].unique())
            fig = px.line(
                median_salary_by_experience,
                x = "work_year",
                y = "median_salary",
                color = "experience_level",
                symbol = "experience_level",
                symbol_map = experience_markers,
                markers=True,
                line_shape = "linear",
                labels = {"work_year": "Work Year", "median_salary": "Median Salary", "experience_level": "Experience Level"})
            fig.update_traces(marker=dict(size=10))
            fig.update_xaxes(dtick = 1, tickformat = "d")
            st.plotly_chart(fig, use_container_width=True)


with tab5:
    st.markdown("<h1 style='color: #006400; text-align: left; font-weight: 700;'> Filters </h1>", 
        unsafe_allow_html=True)
    with st.container():
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            company_size_t6 = st.selectbox("Company Size          ", salaries["company_size"].unique())
            company_pick_t6 = salaries[salaries["company_size"] == company_size_t6]
        with col2:
            select_year_t6 = st.selectbox("Years          ", company_pick_t6["work_year"].unique())
            year_pick_t6 = company_pick_t6[company_pick_t6["work_year"] == select_year_t6]
        with col3:
            select_location_t6 = st.multiselect("Company Locations          ", year_pick_t6["company_location"].unique(),
                                        default = year_pick_t6["company_location"].unique())
            location_pick_t6 = year_pick_t6[year_pick_t6["company_location"].isin(select_location_t6)]
        with col4:
            select_employment_t6 = st.multiselect("Employment Types          ", location_pick_t6["employment_type"].unique(),
                                        default = location_pick_t6["employment_type"].unique())
            employment_pick_t6 = location_pick_t6[location_pick_t6["employment_type"].isin(select_employment_t6)]
        with col5:
            experience_lvl_t6 = st.multiselect("Experience Level           ", employment_pick_t6["experience_level"].unique(),
                                        default = employment_pick_t6["experience_level"].unique())
            year_pick_t6 = employment_pick_t6[employment_pick_t6["experience_level"].isin(experience_lvl_t6)]
    
    
    st.markdown("<h1 style='color: #006400; text-align: left; font-weight: 700;'> How does Median Salary Differ by Categories? </h1>", 
        unsafe_allow_html=True)
    compare_columns = st.radio(
        "Comparing by:",
        ["company_location", "job_title"],
        format_func=lambda value: {
            "company_location": "Company Location",
            "job_title": "Job Title",
        }[value],
        horizontal=True)
    options = year_pick_t6[compare_columns].value_counts().head(20).index.tolist()
    col_a, col_b = st.columns(2)
    group_a = col_a.selectbox(" ", options, index=0)
    group_b = col_b.selectbox(" ", options, index=min(1, len(options) - 1))
    salary_a = year_pick_t6.loc[year_pick_t6[compare_columns] == group_a, "salary_in_usd"]
    salary_b = year_pick_t6.loc[year_pick_t6[compare_columns] == group_b, "salary_in_usd"]
    median_a = salary_a.median()
    median_b = salary_b.median()
    difference_pct = (median_b / median_a - 1) * 100
    # COMPARISON CHART BOX PLOTS
    _, col_m1, col_m2, _ = st.columns([1, 2, 2, 1])
    col_m1.metric(label=f"{group_a}", value=f"${median_a:,.0f}", delta = f"100%", border=True)
    col_m2.metric(label=f"{group_b}", value=f"${median_b:,.0f}", delta=f"{difference_pct:+.1f}% vs {group_a}", border=True)
    fig_vs = px.violin(
        year_pick_t6[year_pick_t6[compare_columns].isin([group_a, group_b])],
        x=compare_columns,
        y="salary_in_usd",
        color=compare_columns,
        color_discrete_map={group_a: "#2a9d8f", group_b: "#e76f51"},
        box=True,          
        points="all",      
        labels={"salary_in_usd": "Salary (USD)", compare_columns: ""})
    fig_vs.update_layout(showlegend=False, yaxis_title="Salary (USD)")
    st.plotly_chart(fig_vs, width="stretch")

    st.markdown("<h1 style='color: #006400; text-align: left; font-weight: 700;'> Which Data Science Professions Earn the Most and the Least? </h1>", 
        unsafe_allow_html=True)
    # BAR CHART HORIZONTAL
    job_medians = (
        year_pick_t6.groupby("job_title", as_index=False)
            .agg(median_salary=("salary_in_usd", "median"), count=("salary_in_usd", "size")))
    top_10_jobs = job_medians.sort_values("median_salary", ascending=False).head(10)
    bottom_10_jobs = job_medians.sort_values("median_salary", ascending=True).head(10)
    with st.container():
        col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Job Titles")
        fig_top = px.bar(
            top_10_jobs.sort_values("median_salary", ascending=True),
            x="median_salary",
            y="job_title",
            orientation="h",
            text_auto=True,
            labels={"median_salary": "Median Salary (USD)", "job_title": ""},
            color="median_salary",
            color_continuous_scale="aggrnyl")
        fig_top.update_layout(showlegend=False, xaxis_title="Median Salary (USD)", yaxis_title="")
        fig_top.update_traces(texttemplate="$%{x:,.0f}", textposition="outside")
        st.plotly_chart(fig_top, use_container_width=True)
    with col2:
        st.subheader("Bottom 10 Job Titles")
        fig_bottom = px.bar(
            bottom_10_jobs.sort_values("median_salary", ascending=True),
            x="median_salary",
            y="job_title",
            orientation="h",
            text_auto=True,
            labels={"median_salary": "Median Salary (USD)", "job_title": ""},
            color="median_salary",
            color_continuous_scale="ylorbr")
        fig_bottom.update_layout(showlegend=False, xaxis_title="Median Salary (USD)", yaxis_title="")
        fig_bottom.update_traces(texttemplate="$%{x:,.0f}", textposition="outside")
        st.plotly_chart(fig_bottom, use_container_width=True)

    st.markdown("<h1 style='color: #006400; text-align: left; font-weight: 700;'> How does Experience Level Differ by Job Position? </h1>", 
        unsafe_allow_html=True)
    #STACKED BAR CHART
    st.subheader("Stacked Barchart of Top 15 Job Titles")
    top_jobs = (year_pick_t6["job_title"].value_counts().head(15).index)
    job_experience_counts = (
        year_pick_t6[year_pick_t6["job_title"].isin(top_jobs)]
        .groupby(["job_title", "experience_level"], as_index=False)
        .size()
        .rename(columns={"size": "count"}))
    fig4 = px.bar(
        job_experience_counts,
        x="job_title",
        y="count",
        text_auto=True,
        text="count",
        color="experience_level",
        barmode="stack",
        title="",
        labels={
            "job_title": "Job Title",
            "count": "Number of Records",
            "experience_level": "Experience Level"})
    st.plotly_chart(fig4, use_container_width=True)


    experience_order = ["EN", "MI", "SE", "EX"]
    work_order = ["On-site", "Hybrid", "Fully remote"]
    category_order = {
        "experience_level": experience_order,
        "company_size": ["S", "M", "L"],

        "remote_label": work_order} 
