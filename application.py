# %%writefile main.py

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import datetime
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
from io import BytesIO
import altair as alt
from datetime import date
import base64
import pickle
from pathlib import Path
import streamlit_authenticator as stauth

hide_menu = """
<style>
#MainMenu {
    visibility:visible;
}
footer {
    visibility:visible;
}
footer:after{
    content:'Copyright © 2022: Indispensables';
    display:block;
    position:relative;
    color:red;
    padding:5px;
    top:3px;
}
</style>
"""

st.markdown(hide_menu, unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg(png_file):
    bin_str = get_base64(png_file)
    page_bg_img ="""
        <style>
        .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: 1390px 750px;           
        }
        </style>
    """ % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Loading dataframe

d1 = pd.read_csv('df1.csv')
d2 = pd.read_csv('df2.csv')
d3 = pd.read_csv('df3.csv')


# Define dataframes
df = pd.concat([d1, d2, d3], axis=0, ignore_index=True)
# df = pd.read_csv('nigeria_election_analysis.csv')

df.dropna(subset=['tweet_clean'], inplace=True)
df['time'] = pd.to_datetime(df['time']).dt.normalize()

# Define time
time = value=datetime.datetime(year=2022, month=6, day=10, hour=16, minute=30)
time = time.date()

# Setting background
set_bg("index.jpeg")

with st.sidebar:
    navigation= option_menu(None, ["Home", "Politics Today", "Presidential Election Prediction"], 
        icons=['house-fill', "book-half", "check-circle-fill"], default_index=1)


if navigation == "Home":
    st.markdown("<h2 style='text-align: center; color: black;'>The Indispensables 2022 Election Analysis </h2>", unsafe_allow_html=True)
    st.markdown("*****************")
    st.subheader("About Elections")
    
    col1, col2 = st.columns(2)


    with col1:
        st.write("""
        Elections are held in Nigeria every four years. The election cycle is upon us
        and at indespensables we aim to keep you up to date with the latest trends, changing popularities of the various political 
        coalitions and political figures as we head towards the election.
        """)    

        st.markdown("*****************")


    with col2:
         st.image("gettyimages-866631268-2048x2048.jpg")   
    
    
    
if navigation == "Politics Today":
    st.write("""
    **_The 2023 Nigerian elections will be held, in large part, on 25 February . Here, we seek to show you the trending topics, the changing popularities
    of political parties and politicians_**
    """)

    navigate2 = option_menu(
    menu_title=None,
    options=["Trending Topics", "Political Parties", "Political Figures"],
    icons=["activity", "flag-fill", "people-fill"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
    "container": {"padding": "0!important"},
    "icon": {"color": "#fff", },
    "nav-link": {"font-size": "12px", "text-align": "left", "margin":"5px", "--hover-color": "#eee"},
    "nav-link-selected": {"background-color": "#A23838"},

    },
    )




    if navigate2 == "Trending Topics":
        st.subheader("Dates of interest")        
        st.write("What dates do you want to get Trending topics in this election cycle?")
        start = st.date_input(label='Start: ', value=datetime.datetime(year=2022, month=5, day=20, hour=16, minute=30), key='#start', help="The start date time", on_change=lambda : None)


        if start < time:
            st.error('Sorry we do not have data for this period. For more objective analysis try inputing dates from 10-06-2022')
        
        # Start date
        end = st.date_input(label='End: ', value=datetime.datetime(year=2022, month=5, day=30, hour=16, minute=30), key='#end', help="The end date time", on_change=lambda : None)

        if start < end and start >= time:
            st.success('Start date: `%s`\n\nEnd date:`%s`' % (start, end))
            start = start.strftime("%Y-%m-%d")
            end = end.strftime("%Y-%m-%d")
            
            # Greater than the start date and smaller than the end date
            mask = (df['time'] > start) & (df['time'] <= end)

            # Applying filter
            df1 = df.loc[mask]

            # st.subheader("Top trends on social media in the run up to the 2022 elections")
            choice = st.sidebar.radio(label = "Go To:", options = ["Top trends on social media in the run up to the 2022 elections", 'Polarity of sentiments of the electorate heading towards the general election'])
            
            if choice == "Top trends on social media in the run up to the 2022 elections":
                st.subheader("Top trends on social media in the run up to the 2022 elections")
                    
                # Converting text descriptions into vectors using TF-IDF using Bigram
                tf = TfidfVectorizer(ngram_range=(2, 2), stop_words='english', lowercase = False)
                tfidf_matrix = tf.fit_transform(df1['tweet_clean'])
                total_words = tfidf_matrix.sum(axis=0) 
            
                # Finding the word frequency
                freq = [(word, total_words[0, idx]) for word, idx in tf.vocabulary_.items()]
                freq =sorted(freq, key = lambda x: x[1], reverse=True)

                # converting into dataframe 
                bigram = pd.DataFrame(freq)
                bigram.rename(columns = {0:'bigram', 1: 'count'}, inplace = True) 

                
                # Taking first 20 records
                popular_words  = bigram.head(20)
                popular_words['count'] = ((popular_words['count']/ popular_words['count'].sum()) * 100).round(2)

                # List of utems to be plotted
                names = popular_words['bigram'].to_list()
                counts = popular_words['count'].to_list()

                # Dataframe source of items
                source = pd.DataFrame({
                    'counts': counts,
                    'names': names
                    })

                # attempting to sort plot
                source1 = source.sort_values(['counts'], ascending=[False])

                # Bar chart
                bar_chart = alt.Chart(source1).mark_bar().encode(
                    y='counts',
                    x='names',
                    color = 'names'
                    )

                # Labels of bar chart
                text = bar_chart.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3  # Nudges text to right so it doesn't appear on top of the bar
                    ).encode(text='counts')

                # Plot
                plt = (bar_chart + text).properties(height=600)
                st.altair_chart(plt, use_container_width=True)
                
            if choice == 'Polarity of sentiments of the electorate heading towards the general election':
                st.subheader('Polarity of sentiments of the electorate heading towards the general election') 
                # st.sidebar.subheader(' Polarity of sentiments of the electorate heading towards the general election') 

                st.write("""
                **_The line chart represents the changes in polarity of the electorate as we head toward the 2022
                election. Score of 1 represents very positive, -1 represents very negative, and 0 represents 
                neautral sentiments_** 
                """)

                # applying filter
                df2 = df.loc[mask]

                # Selecting desired attributes and grouping like items in record
                polarity = df2[['time', 'Polarity']]
                polarity = polarity.groupby('time',  as_index=False, sort=False).agg({'Polarity': 'mean'})

                # getting a list of the time and polarty from dataframe
                date1 = polarity['time'].to_list()
                polaritys = polarity['Polarity'].to_list()

                # Source dataframe
                source = pd.DataFrame({
                    'polaritys': polaritys,
                    'date': date1
                    })
                
                # Line chart
                line_chart = alt.Chart(source).mark_line().encode(
                    y='polaritys',
                    x='date',
                    )

                # Labellling the line chart
                text = line_chart.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3  # Nudges text to right so it doesn't appear on top of the bar
                    )

                # plot
                plt = (line_chart + text).properties(height=600)
                st.altair_chart(plt, use_container_width=True)

        else:
            st.error('Error: End date must fall after start date.') 

    if navigate2 == "Political Parties":        
        st.write("What dates do you want to get popularity of various political parties?")
        
        # start date
        start = st.date_input(label='Start: ', value=datetime.datetime(year=2022, month=5, day=20, hour=16, minute=30), key='#start', help="The start date time", on_change=lambda : None)


        if start < time:
            st.error('Sorry we do not have data for this period. For more objective analysis try inputing dates from 10-06-2022')
        
        # End date
        end = st.date_input(label='End: ', value=datetime.datetime(year=2022, month=5, day=30, hour=16, minute=30), key='#end', help="The end date time", on_change=lambda : None)


        if start < end and start >= time:
            st.success('Start date: `%s`\n\nEnd date:`%s`' % (start, end))
            start = start.strftime("%Y-%m-%d")
            end = end.strftime("%Y-%m-%d")

            #greater than the start date and smaller than the end date
            mask = (df['time'] > start) & (df['time'] <= end)

            # Applying filter
            df1 = df.loc[mask]

            choice2 = st.sidebar.radio(label = "Go To:", options = ['Social Media Popularity of Political Parties heading towards the general election', 'Polarity of sentiments of the electorate heading towards the general election'])
            if choice2 == 'Social Media Popularity of Political Parties heading towards the general election':

                # Selecting columns of interest, dropping unwanted records, and grouping like records in dataframe
                words_partys = df1[['Party']]
                words_party = words_partys.groupby(['Party'])['Party'].count().reset_index(name='count')
                words_party.drop(words_party.loc[words_party['Party']== 'None'].index, inplace=True)
                words_party['count'] = ((words_party['count']/ words_party['count'].sum()) * 100).round(2)


                # Names and counts
                names = words_party['Party'].to_list()
                counts = words_party['count'].to_list()
                
                # Dataframe source
                source = pd.DataFrame({
                    'counts': counts,
                    'names': names
                    })

                # Sorting 
                source1 = source.sort_values(['counts'], ascending=[False])

                # Plot
                bar_chart = alt.Chart(source1).mark_bar().encode(
                    y='counts',
                    x='names',
                    color = 'names',
                    )

                # Labels
                text = bar_chart.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3  # Nudges text to right so it doesn't appear on top of the bar
                    ).encode(text='counts')
            
                plt = (bar_chart + text).properties(height=600)
                st.altair_chart(plt, use_container_width=True)

            if choice2 == 'Polarity of sentiments of the electorate heading towards the general election':
            
                st.write("""
                The line chart represents the changes in polarity of the electorate as we head toward the 2022
                election. Score of 1 represents very positive, -1 represents very negative, and 0 represents 
                neautral sentiments
                """)
                
                # Applying filter
                df2 = df.loc[mask]

                # Seelecting necessary attributes
                words_partys= df2[['Party', 'time', 'Polarity']]

                # Applying filter 
                words_party = words_partys.groupby(['time','Party'], as_index=False, sort=False).agg({'Polarity': 'mean'})
                words_party.drop(words_party.loc[words_party['Party']== 'None'].index, inplace=True)

                # Getting lists of the varous attributes in word Partys dataframe
                date1 = words_party['time'].to_list()
                polaritys = words_party['Polarity'].to_list()
                party = words_party['Party'].to_list()

                # Source daatframe
                source = pd.DataFrame({
                    'polaritys': polaritys,
                    'date': date1,
                    'symbol': party
                    })

                # Line chart
                line_chart = alt.Chart(source).mark_line().encode(
                    y='polaritys',
                    x='date',
                    color='symbol',
                    strokeDash='symbol',
                    )

                # Labelling
                text = line_chart.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3  # Nudges text to right so it doesn't appear on top of the bar
                    )
            
                # Plot
                plt = (line_chart + text).properties(height=600)
                st.altair_chart(plt, use_container_width=True)


            # Date else condition
        else:
            st.error('Error: End date must fall after start date.')       




    if navigate2 == "Political Figures":        
        st.write("What dates do you want to get popularity of various political figures?")

        # Start date
        start = st.date_input(label='Start: ', value=datetime.datetime(year=2022, month=5, day=20, hour=16, minute=30), key='#start', help="The start date time", on_change=lambda : None)

        if start < time:
            st.error('Sorry we do not have data for this period. For more objective analysis try inputing dates from 10-06-2022')
        
        # End date
        end = st.date_input(label='End: ', value=datetime.datetime(year=2022, month=5, day=30, hour=16, minute=30), key='#end', help="The end date time", on_change=lambda : None)

        if start < end and start >= time:
            st.success('Start date: `%s`\n\nEnd date:`%s`' % (start, end))
            start = start.strftime("%Y-%m-%d")
            end = end.strftime("%Y-%m-%d")
            
            # Greater than the start date and smaller than the end date
            mask = (df['time'] > start) & (df['time'] <= end)

            # Applying filter
            df1 = df.loc[mask]

            choice3 = st.sidebar.radio(label = "Go To:", options = ["Social Media Popularity of Political Figures heading towards the general election", "Polarity of sentiments towards various presidential aspirants for Kenyas 2022 general election"])
            if choice3 == "Social Media Popularity of Political Figures heading towards the general election":

                # Selecting records, attributes of intrest and dropping unwanted attributes
                words_presidents = df1[['presidential_aspirant']]
                words_president = words_presidents.groupby(['presidential_aspirant'])['presidential_aspirant'].count().reset_index(name='count')
                words_president.drop(words_president.loc[words_president['presidential_aspirant']== 'None'].index, inplace=True)
                words_president['count'] = ((words_president['count']/ words_president['count'].sum()) * 100).round(2)

                # Names and counts
                names = words_president['presidential_aspirant'].to_list()
                counts = words_president['count'].to_list()

                # Source dataframe
                source = pd.DataFrame({
                    'counts': counts,
                    'names': names
                    })

                # Sorting
                source1 = source.sort_values(['counts'], ascending=[False])

                # Bar chart
                bar_chart = alt.Chart(source1).mark_bar().encode(
                    y='counts',
                    x='names',
                    color = 'names',
                    )

                # labelling bar chart
                text = bar_chart.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3  # Nudges text to right so it doesn't appear on top of the bar
                    ).encode(text='counts')

                # Plotting bar chart
                plt = (bar_chart + text).properties(height=600)
                st.altair_chart(plt, use_container_width=True)

            if choice3 == "Polarity of sentiments towards various presidential aspirants for Kenyas 2022 general election":
            
                st.write("""
                The line chart represents the changes in polarity of the electorate as we head toward the 2022
                election. Score of 1 represents very positive, -1 represents very negative, and 0 represents 
                neautral sentiments 
                """)
            
                # Applying filter
                df2 = df.loc[mask]

                # selectring attributes of interest, dropping unwanted records, and grouping records based on time and presidential aspirnat
                words_presidents = df2[['presidential_aspirant', 'time', 'Polarity']]
                words_president = words_presidents.groupby(['time','presidential_aspirant'], as_index=False, sort=False).agg({'Polarity': 'mean'})
                words_president.drop(words_president.loc[words_president['presidential_aspirant']== 'None'].index, inplace=True)

                # Converting the date, polarity ans presidential aspirant attributes in dataframe to lists
                date1 = words_president['time'].to_list()
                polaritys = words_president['Polarity'].to_list()
                aspirant = words_president['presidential_aspirant'].to_list()

                # Source dataframe
                source = pd.DataFrame({
                    'polaritys': polaritys,
                    'date': date1,
                    'symbol': aspirant
                    })
                
                # Line chart
                line_chart = alt.Chart(source).mark_line().encode(
                    y='polaritys',
                    x='date',
                    color='symbol',
                    strokeDash='symbol',
                    )
                
                # Labelling plot
                text = line_chart.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3  # Nudges text to right so it doesn't appear on top of the bar
                    )

                # plotting
                plt = (line_chart + text).properties(height=600)
                st.altair_chart(plt, use_container_width=True)

        else:
            st.error('Error: End date must fall after start date.')




if navigation == "Presidential Election Prediction":
    names = ["Admin 1", "Admin 2"]
    usernames = ["admin1", "admin2"]

    file_path = Path(__file__).parent / "hashed_pw.pkl"
    with file_path.open("rb") as file:
        hashed_passwords = pickle.load(file)

    authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "Elections_Predictor", "abcdef", cookie_expiry_days=30)
    name, authentication_status, username = authenticator.login("Login Section", "main")

    if authentication_status == False:
        st.error("Username/password is incorrect")

    if authentication_status == None:
        st.warning("Please Enter your username and password")

    if authentication_status:
        authenticator.logout("Logout", "main")
        st._main.title(f"Welcome {name}")         

        
        st.write("At indespensable we try to predict the presidential aspirant most likely to win the forthcoming August 9th, 2022 elections. This prediction is made using various sentiments obtained from social media regarding the general election. YOu can observe the changes in favour of the presidential aspirants for different periods leading towards the election")
        st.write("What dates do you want to get popularity of various political figures?")
        
        # start date
        start = st.date_input(label='Start: ', value=datetime.datetime(year=2022, month=5, day=20, hour=16, minute=30), key='#start', help="The start date time", on_change=lambda : None)


        if start < time:
            st.error('Sorry we do not have data for this period. For more objective analysis try inputing dates from 10-06-2022')

        # End date  
        end = st.date_input(label='End: ', value=datetime.datetime(year=2022, month=5, day=30, hour=16, minute=30), key='#end', help="The end date time", on_change=lambda : None)


        if start < end and start >= time:
            st.success('Start date: `%s`\n\nEnd date:`%s`' % (start, end))
            start = start.strftime("%Y-%m-%d")
            end = end.strftime("%Y-%m-%d")
                
            # Greater than the start date and smaller than the end date
            mask = (df['time'] > start) & (df['time'] <= end)

            # Applying filter
            df1 = df.loc[mask]
           
            # Function to find percentage of positive and negative sentiments
            def pol_percent(subset,total):
                neg_percent = ((subset.groupby('Expressions').count())['Polarity'][0]/total)*100
                pos_percent = ((subset.groupby('Expressions').count())['Polarity'][1]/total)*100

                return neg_percent,pos_percent

            # Dropping records without mentions of any of the presidential aspirants
            df1.drop(df1.loc[df1['presidential_aspirant']== 'None'].index, inplace=True)

            # Getting total number neutral sentiments in the dataframe
            neutral_total = len(df1[df1['Polarity']==0])/3

            # Selecting records from df1 with mentions of various presidential aspirants
            df_peterObi = df1 [df1 ['presidential_aspirant'] == 'peterObi']
            df_Tinubu = df1 [df1 ['presidential_aspirant'] == 'Tinubu']
            df_ATIKU = df1 [df1 ['presidential_aspirant'] == 'ATIKU']

            # Dropping neutral records for the three presidnetial aspirants
            df_peterObi.drop((df_peterObi[df_peterObi['Polarity']==0]).index, inplace=True)
            df_Tinubu.drop((df_Tinubu[df_Tinubu['Polarity']==0]).index, inplace=True)
            df_ATIKU.drop((df_ATIKU[df_ATIKU['Polarity']==0]).index, inplace=True)

            # Getting the total number of records where the three presidential aspirants have been mentioned
            records_peterObi = len(df_peterObi)
            records_Tinubu = len(df_Tinubu)
            records_ATIKU = len(df_ATIKU)

            # Total records with presidential mentions
            total_records = records_peterObi + records_Tinubu + records_ATIKU + neutral_total

            # Finding percentage negative and positive sentiments per presidential aspirnat
            peterObi_total_percent = pol_percent(df_peterObi,records_peterObi)
            Tinubu_total_percent = pol_percent(df_Tinubu,records_Tinubu)
            ATIKU_total_percent = pol_percent(df_ATIKU,records_ATIKU)

            # Finding the favour of presidential aspirants an undecded voters in electorate
            peterObi_pos = (peterObi_total_percent[1] + (Tinubu_total_percent[0] + ATIKU_total_percent[0])/2) * (records_peterObi/total_records )
            Tinubu_pos =(Tinubu_total_percent[1] + (peterObi_total_percent[0] + ATIKU_total_percent[0])/2) * (records_Tinubu/total_records )
            ATIKU_pos = (ATIKU_total_percent[1] + (peterObi_total_percent[0] + Tinubu_total_percent[0])/2) * (records_ATIKU/total_records)
            undecided_pos_percent = neutral_total/total_records * 100

            # Lists of percentage polling of aspirants and undecided voters together with attached labels
            counts = [peterObi_pos, Tinubu_pos, ATIKU_pos, undecided_pos_percent]
            names =  ['peterObi\'s Favour' ,'Tinubu\'s Favour','ATIKU\'s Favour', 'Undecided Voters']

            # Source dataframe
            source = pd.DataFrame({
                'counts': counts,
                'names': names
                })

            # Sorting the counts in ascending order
            source1 = source.sort_values(['counts'], ascending=[False])

            # Bar chart
            bar_chart = alt.Chart(source1).mark_bar().encode(
                y='counts',
                x='names',
                color = 'names'
                )

            # labels for bar chart
            text = bar_chart.mark_text(
                align='left',
                baseline='middle',
                dx=3  # Nudges text to right so it doesn't appear on top of the bar
                ).encode(text='counts')

            # Plot
            plt = (bar_chart + text).properties(height=600)
            st.altair_chart(plt, use_container_width=True)



            
            # Prediction without factoring in lack of undecided voters
            st.subheader('Presidential prediction without factoring in undecided voters')
            st.write("Here we try to predict  the presidential race assuming all the social media who had neutral sentiments did not participate in the forthcoming elections due to voter apathy")

            # Getting the total number of records where the three presidential aspirants have been mentioned
            records_peterObi = len(df_peterObi)
            records_Tinubu = len(df_Tinubu)
            records_ATIKU = len(df_ATIKU)

            # Total without undecided voters
            total = records_peterObi + records_Tinubu + records_ATIKU

            # Finding percentage negative and positive sentiments per presidential aspirnat
            peterObi_total_percent = pol_percent(df_peterObi,records_peterObi)
            raila_total_percent = pol_percent(df_raila,records_raila)
            wajackoyah_total_percent = pol_percent(df_wajackoyah,records_wajackoyah)

            # Finding the favour of presidential aspirants an undecded voters in electorate
            ruto_pos = (ruto_total_percent[1] + (raila_total_percent[0] + wajackoyah_total_percent[0])/2) * (records_ruto/total)
            raila_pos =(raila_total_percent[1] + (ruto_total_percent[0] + wajackoyah_total_percent[0])/2) * (records_raila/total)
            wajackoyah_pos = (wajackoyah_total_percent[1] + (ruto_total_percent[0] + raila_total_percent[0])/2) * (records_wajackoyah/total)

            # Lists of percentage polling of aspirants and undecided voters together with attached labels
            counts = [ruto_pos, raila_pos, wajackoyah_pos]
            names =  ['ruto\'s Favour' ,'raila\'s Favour','wajackoyah\'s Favour']

            # Source dataframe
            source = pd.DataFrame({
                'counts': counts,
                'names': names
                })

            # Sorting the counts in ascending order
            source1 = source.sort_values(['counts'], ascending=[False])

            # Bar chart
            bar_chart = alt.Chart(source1).mark_bar().encode(
                y='counts',
                x='names',
                color = 'names',
                )

            # labels for bar chart
            text = bar_chart.mark_text(
                align='left',
                baseline='middle',
                dx=3  # Nudges text to right so it doesn't appear on top of the bar
                ).encode(text='counts')

            #
 
            plt = (bar_chart + text).properties(height=600)
            st.altair_chart(plt, use_container_width=True)


        else:
            st.error('Error: End date must fall after start date.')
