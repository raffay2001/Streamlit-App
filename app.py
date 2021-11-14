import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
from autoscraper import AutoScraper
import pandas as pd
import streamlit as st

header = st.container()
model_training = st.container()
feature = st.container()
with header:
    st.title('Amazon Product Reviews Analysis')
    st.text('SELECT ANY E-COMMERCE WEBSITE')
    AMAZON = st.button("AMAZON")
    # DARAZ  = st.button("DARAZ")

# if DARAZ == True:
#     with model_training:
#         st.header('Drop the link')

#         sel_col, disp_col = st.columns(2)
#         n_estimators = sel_col.text_input('Please drop the link of product reviews to see the analysis','https://www.daraz.pk/products/p47-wireless-headphones-bluetooth-stereo-head-phones-foldable-headset-with-mic-wireless-built-in-mic-compaible-for-all-android-devices-and-pc-i220638343-s1434420364.html?spm=a2a0e.searchlist.list.7.717d40200AoMv2&search=1')

if AMAZON == True:
    with model_training:
        st.header('Drop the link')

        sel_col, disp_col = st.columns(2)
        n_estimators = sel_col.text_input('Please drop the link of product reviews to see the analysis','https://www.amazon.com/Maybelline-SuperStay-Matte-Liquid-Lipstick/product-reviews/B074VD4X86/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews')

def scrape(x):
    amazon_url = "https://www.amazon.com/Maybelline-SuperStay-Matte-Liquid-Lipstick/product-reviews/B074VD4X86/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"

    wish_list = ['Love it!', '362 people found this helpful']

    scraper_1 = AutoScraper()
    result = scraper_1.build(amazon_url, wish_list)
    review_title_key = scraper_1.get_result_similar(amazon_url, grouped=True)
    keys = list(review_title_key.keys())
    title_key = keys[0]
    scraper_1.set_rule_aliases({title_key: 'Review_Title'})
    scraper_1.keep_rules([title_key])
    scraper_1.save('Amazon-Reviews')
    scraper_1.load('Amazon-Reviews')
    link = x
    result_1 = []
    for i in range(1, 20):
        result = scraper_1.get_result_similar(
            link+'&pageNumber='+str(i), group_by_alias=True)
        result_1.append(result['Review_Title'])
    reviews = []
    for i in range(len(result_1)):
        for j in range(len(result_1[i])):
            reviews.append(result_1[i][j])
    review_dict = {'prodct_review': reviews}
    df = pd.DataFrame(review_dict)
    df.to_csv('product_reviews.csv')
    return df

def scrape_DARAZ(x):
    DARAZ_url = "https://www.daraz.pk/products/p47-wireless-headphones-bluetooth-stereo-head-phones-foldable-headset-with-mic-wireless-built-in-mic-compaible-for-all-android-devices-and-pc-i220638343-s1434420364.html?spm=a2a0e.searchlist.list.7.717d40200AoMv2&search=1"

    wish_list = ['𝐸𝑝𝑖𝑐 𝑡ℎ𝑖𝑛𝑔 𝑞𝑢𝑎𝑙𝑖𝑡𝑦 𝑝𝑒𝑟 𝑟ℎ𝑜𝑟𝑎 𝑘𝑎𝑎𝑚 ℎ𝑎 𝑏𝑢𝑡 𝑎𝑤𝑎𝑧 𝑛 𝑏𝑒𝑎𝑡 𝑙𝑖𝑘𝑒 𝑎 𝑏𝑜𝑜𝑚𝑏𝑜𝑥 𝑝𝑎𝑐𝑘𝑖𝑛𝑔 𝑤𝑎𝑠 𝑔𝑜𝑜𝑑 100% 𝑟𝑒𝑙𝑖𝑎𝑏𝑙𝑒 𝑠𝑒𝑙𝑙𝑒𝑟', 'Color Family:وائن سرخ']

    scraper_1 = AutoScraper()
    result = scraper_1.build(DARAZ_url, wish_list)
    review_title_key = scraper_1.get_result_similar(DARAZ_url, grouped=True)
    keys = list(review_title_key.keys())
    title_key = keys[0]
    scraper_1.set_rule_aliases({title_key: 'Review_Title'})
    scraper_1.keep_rules([title_key])
    scraper_1.save('DARAZ-Reviews')
    scraper_1.load('DARAZ-Reviews')
    link = x
    result_1 = []
    for i in range(1, 10):
        result = scraper_1.get_result_similar(
            link+str(i), group_by_alias=True)
        result_1.append(result['Review_Title'])
    reviews = []
    for i in range(len(result_1)):
        for j in range(len(result_1[i])):
            reviews.append(result_1[i][j])
    review_dict = {'prodct_review': reviews}
    df = pd.DataFrame(review_dict)
    df.to_csv('product_reviews.csv')
    return df

if AMAZON == True:
    scrape(n_estimators)

# if DARAZ == True:
#     scrape_DARAZ(n_estimators)

nltk.download('vader_lexicon')


def parse_values(x):
    if x < 0:
        return 'negative'
    elif x == 0:
        return 'neutral'
    else:
        return 'positive'

# if AMAZON == True or DARAZ == True:
if AMAZON == True:
    sid = SentimentIntensityAnalyzer()
    df1 = pd.read_csv('product_reviews.csv')
    df1['prodct_review'].isnull().sum()
    df1['scores'] = df1['prodct_review'].apply(
        lambda review: sid.polarity_scores(review))
    df1['compound'] = df1['scores'].apply(lambda score_dict: score_dict['compound'])
    df1['comp_score'] = df1['compound'].apply(parse_values)
    df1.to_csv('sentiments_1.csv')

    with feature:
        st.subheader('Bar Charts for Sentiment Analysis')
        pos_neg = df1['comp_score'].value_counts()
        st.bar_chart(pos_neg)
        st.subheader('Line Charts for Sentiment Analysis')
        pos_neg = df1['comp_score'].value_counts()
        st.line_chart(pos_neg)
        selected_element = st.selectbox("Pick a plot",('BarChart','LineChart'))
        st.subheader(selected_element+'for Sentiment Analysis')
        comp = df1['compound'].value_counts()
        if selected_element == 'BarChart':
            st.bar_chart(comp)
        else:
            st.line_chart(comp)
