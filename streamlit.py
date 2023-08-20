import streamlit as st
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# Change working directory
# os.chdir(r"C:\Users\User\Documents\GitHub\UBD\SS4290 Project")
os.chdir('/Users/faiz/Desktop/UBD Final Year/UBD/SS4290 Project') # CHANGE DIRECTORY TO CURRENT DIRECTORY


# Load model and vectorizer
svm_model = pickle.load(open('project_data/svm_model.pkl', 'rb'))
vectorizer = pickle.load(open('project_data/vectorizer.pkl', 'rb'))

# Predict single complaint
def single():
    with st.form("Input"):
        queryText = st.text_area("Enter a complaint", height=200)
        btnRun = st.form_submit_button('Run')
        
    if btnRun:    
        # run query
        single_predict(queryText)

        
def single_predict(text_input):
    vect = vectorizer.transform([text_input])
    pred = svm_model.predict(vect)
    results = svm_model.predict_proba(vect)

    if text_input == '':
        pass
    else:
        st.markdown(f'Predicted category: **{get_key(pred)}**')
        prob_per_class_dictionary = dict(zip(svm_model.classes_, results))
        pie_chart = drawPie(prob_per_class_dictionary[0])
        st.pyplot(pie_chart)


# Predict multiple complaint
def multi():
    
    uploaded_file = st.file_uploader("Please upload an Excel file", type=['csv', 'xlsx'])
    
    # Process uploaded file
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Predicted Category'] = None
        nrows = len(df.index)
        for index in df.index:
            vect = vectorizer.transform([df['Complaint Narrative'][index]])
            pred = svm_model.predict(vect)
            # results = svm_model.predict_proba(vect)
            df['Predicted Category'][index] = get_key(pred)
        
        st.success('Finished processing the uploaded file!')
        
        df_val_counts = df['Predicted Category'].value_counts(dropna = True)
        result_summary = df_val_counts.reset_index()
        result_summary.columns = ['Predicted Category', 'Count'] # change column names
        
        result_dict = {}
        for index in result_summary.index:
            result_dict[result_summary["Predicted Category"][index]] = result_summary["Count"][index]
        
        predicted_categories = list(result_dict.keys())
        counts = list(result_dict.values())

        # Create bar plot
        colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'wheat']
        
        plt.bar(predicted_categories, counts, color = colors, width = 0.4)

        plt.xticks(rotation=90)
        plt.xlabel("Predicted Category", fontweight='bold')
        plt.ylabel("No. of Complaints", fontweight='bold')
        plt.title("Results Summary", fontweight='bold')
        
        st.pyplot(plt)
        
        # Create table
        result_summary.rename(columns = {'Count' : 'Count (Total: ' + str(nrows) + ')'}, inplace = True)
        st.table(result_summary)
        
        # Export data
        ts = str(datetime.datetime.now().timestamp())
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
           "Export to CSV",
           csv,
           "predicted_" + ts + ".csv",
           "text/csv",
           key='download-csv'
        )
    
    else:
        st.warning("You need to upload an Excel file.")
        

# Draw piechart of probability of predicted classes by model
def drawPie(values):
    sizes = []
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'wheat']
    for v in values:
        sizes.append("{:.1f}".format(v*100))
    labels = [f'Credit Reporting ({sizes[0]}%)', f'Debt Collection ({sizes[1]}%)',
              f'Mortgages and Loans ({sizes[2]}%)', f'Credit Card ({sizes[3]}%)', f'Retail Banking ({sizes[4]}%)']
    patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    plt.legend(patches, labels, loc="best")
    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')
    plt.title("Probability of complaint category", bbox={'facecolor':'0.8', 'pad':5})
    plt.tight_layout()
    # fig1, ax1 = plt.subplots()
    # ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    return plt


# Convert numbered category to text
def get_key(val):
    category_dict = {'Credit Reporting': 0, 'Debt Collection': 1, 'Mortgages and Loans': 2,
                     'Credit Card': 3, 'Retail Banking': 4}

    for key, value in category_dict.items():
        if val == value:
            return key

    return "key doesn't exist"


def main():

    st.set_page_config(
        page_title="Complaint Classification",
    )

    # Sidebar
    st.sidebar.image("project_data/ubd_logo.png")
    st.sidebar.title(
        'Summarizing Business Reporting Collection using Machine Learning')
    option = st.sidebar.selectbox(
        'Predict complaint:',
        ('Single', 'Multiple')
    )

    # Center image on sidebar
    st.markdown(
        """
        <style>
            [data-testid=stSidebar] [data-testid=stImage]{
                text-align: center;
                display: block;
                margin-left: auto;
                margin-right: auto;
                width: 100%;
            }
        </style>
        """, unsafe_allow_html=True
    )

    if option == 'Single' or option == '':
        single()
    elif option == 'Multiple':
        multi()


if __name__ == '__main__':
    main()
