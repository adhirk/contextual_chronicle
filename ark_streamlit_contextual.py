import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import datetime  # Import the datetime module

# Load the saved model, vectorizer, and label encoder
model_filename = 'logistic_regression_model.pkl'
vectorizer_filename = 'tfidf_vectorizer_lem.pkl'
label_encoder_filename = 'label_encoder.pkl'

@st.cache
with open(model_filename, 'rb') as model_file:
    loaded_classifier = pickle.load(model_file)

@st.cache
with open(vectorizer_filename, 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

@st.cache
with open(label_encoder_filename, 'rb') as label_encoder_file:
    loaded_label_encoder = pickle.load(label_encoder_file)

# Set the page title and configure the page
st.set_page_config(
    page_title="Contextual Chronicle",
    page_icon=":newspaper:",
    layout="wide",
)

# Create Streamlit app title with custom CSS style for font and lines
st.markdown(
    '''
    <div style="text-align: center;">
        <span style="font-family: Abril Fatface; font-size: 108px;">Contextual Chronicle</span><br>
        <p style="font-size: 20px; margin-bottom: -10px;">By Adhitan Rani Kasinathan</p>  <!-- Add the author's name -->
        <span style="border-top: 2px solid #000; width: 60%; margin-top: 10px; display: inline-block;"></span><br>
        <span style="height: 5px; display: block;"></span>  <!-- Add 20px space -->
    </div>
    ''',
    unsafe_allow_html=True
)

# Create the fixed box for menu items with options, increased spacing, and space below
box_style = """
    border: 1px solid #ffffff;
    padding: 10px;
    border-radius: 5px;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 50px; /* Add 50px space below the menu box */
    display: flex;
    flex-direction: column; /* Display menu items vertically */
    align-items: center; /* Center align menu items horizontally */
    background-color: white; /* Set the background color of the box to white */
"""

fixed_box = f'<div style="{box_style}">'
menu_items = [
    ("Politics", "https://your-politics-url.com"),
    ("Travel", "https://your-travel-url.com"),
    ("Sports", "https://your-sports-url.com"),
    ("Business & Finance", "https://your-business-url.com"),
    ("Food & Drink", "https://your-food-url.com"),
    ("Style & Beauty", "https://your-beauty-url.com"),
    ("Home & Living", "https://your-living-url.com"),
    ("Wellness", "https://your-wellness-url.com"),
    ("Parenting", "https://your-parenting-url.com")
]
menu_items_html = ''.join(
    f'<a href="{item[1]}" style="text-decoration: none; padding: 10px 25px; background-color: black; color: white; margin: 5px; border-radius: 5px;">{item[0]}</a>'
    for item in menu_items
)
fixed_box += f'<div>{menu_items_html}</div>'
fixed_box += '</div>'

# Display the menu items box
st.markdown(fixed_box, unsafe_allow_html=True)

# Add dynamic date and time below the menu box
# current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# st.write("Current Date and Time:", current_datetime)


# # Add dynamic date and time below the menu box in the desired format
# current_datetime = datetime.datetime.now().strftime("%b %d, %Y, %I:%M %p")
# formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# st.write(current_datetime)

# Sidebar for user input
st.sidebar.title("User Input")

input_headline = st.sidebar.text_input("Enter the headline:")
input_article = st.sidebar.text_area("Enter the article text:")

space_pixels = 430  # Specify the number of pixels for space

if st.sidebar.button("Predict"):
    input_sentence = input_headline + ' ' + input_article
    X_new_data_transformed = loaded_vectorizer.transform([input_sentence])
    y_pred_new = loaded_classifier.predict(X_new_data_transformed)
    predicted_class_name = loaded_label_encoder.inverse_transform(y_pred_new)[0]

    # Push the "Predicted Results" with specified space (in pixels) below the "Predict" button
    st.sidebar.markdown(
        f'<div style="height: {space_pixels}px;"></div>',
        unsafe_allow_html=True,
    )


    # Create a box for the "Predicted Results" section
    st.sidebar.markdown(
        f"""
        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px;">
            <h3>Predicted Results</h3>
            <p>Show Ads for: <strong>{predicted_class_name}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # # Create a container to hold the "Predicted Results" section
    # prediction_container = st.container()

    # Create two equal-sized columns
    col1, col2 = st.columns(2)

    # Display predictions in the left column
    with col1:
        st.markdown(
            f'<div style="font-family: Abril Fatface; font-size: 24px;">{input_headline}</div>',
            unsafe_allow_html=True
        )
        st.write("<div style='height: 1px;'></div>", unsafe_allow_html=True)
        current_datetime = datetime.datetime.now().strftime("%b %d, %Y, %I:%M %p")
        formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.write(current_datetime)

        st.write(input_article)  # Display the article text without "Headline" text

    # # Create a container to hold the "Predicted Results" section
    # prediction_container = st.container()

    # Display predicted images in the right column with adjusted size
    with col2:
        # Display an image based on the predicted class name
        if predicted_class_name == "ARTS & CULTURE":
            travel_image = Image.open("ARTS & CULTURE.jpg")
            st.image(travel_image, caption="Arts & Culture", use_column_width=True)

        # Display an image based on the predicted class name
        if predicted_class_name == "BUSINESS & FINANCES":
            travel_image = Image.open("BUSINESS & FINANCES.jpg")
            st.image(travel_image, caption="Business & Finances", use_column_width=True)
    
        # Display an image based on the predicted class name
        if predicted_class_name == "COMEDY":
            travel_image = Image.open("COMEDY.jpg")
            st.image(travel_image, caption="Comedy", use_column_width=True)

        # Display an image based on the predicted class name
        if predicted_class_name == "CRIME":
            travel_image = Image.open("CRIME.jpg")
            st.image(travel_image, caption="Crime", use_column_width=True)
            
        # Display an image based on the predicted class name
        if predicted_class_name == "EDUCATION":
            travel_image = Image.open("EDUCATION.jpg")
            st.image(travel_image, caption="Education", use_column_width=True)

        # Display an image based on the predicted class name
        if predicted_class_name == "ENTERTAINMENT":
            travel_image = Image.open("ENTERTAINMENT.jpg")
            st.image(travel_image, caption="Entertainment", use_column_width=True)

        # Display an image based on the predicted class name
        if predicted_class_name == "ENVIRONMENT":
            travel_image = Image.open("ENVIRONMENT.jpg")
            st.image(travel_image, caption="Environment", use_column_width=True)
            
        # Display an image based on the predicted class name
        if predicted_class_name == "FOOD & DRINK":
            travel_image = Image.open("FOOD & DRINK.jpg")
            st.image(travel_image, caption="Food & Drink", use_column_width=True)

        # Display an image based on the predicted class name
        if predicted_class_name == "GROUPS VOICES":
            travel_image = Image.open("GROUPS VOICES.jpg")
            st.image(travel_image, caption="Group Voices", use_column_width=True)

        # Display an image based on the predicted class name
        if predicted_class_name == "HOME & LIVING":
            travel_image = Image.open("HOME & LIVING.jpg")
            st.image(travel_image, caption="Home & Living", use_column_width=True)

        # Display an image based on the predicted class name
        if predicted_class_name == "MISCELLANEOUS":
            travel_image = Image.open("MISCELLANEOUS.jpg")
            st.image(travel_image, caption="Miscellaneous", use_column_width=True)

        # Display an image based on the predicted class name
        if predicted_class_name == "PARENTING":
            travel_image = Image.open("PARENTING.jpg")
            st.image(travel_image, caption="Parenting", use_column_width=True)

        # Display an image based on the predicted class name
        if predicted_class_name == "POLITICS":
            travel_image = Image.open("POLITICS.jpg")
            st.image(travel_image, caption="Politics", use_column_width=True)
            
        # Display an image based on the predicted class name
        if predicted_class_name == "RELIGION":
            travel_image = Image.open("RELIGION.jpg")
            st.image(travel_image, caption="Religion", use_column_width=True)

        # Display an image based on the predicted class name
        if predicted_class_name == "SCIENCE & TECH":
            travel_image = Image.open("SCIENCE & TECH.jpg")
            st.image(travel_image, caption="Science & Tech", use_column_width=True)

        # Display an image based on the predicted class name
        if predicted_class_name == "SPORTS":
            sports_image = Image.open("SPORTS.jpg")
            st.image(sports_image, caption="Sports", use_column_width=True)

        # Display an image based on the predicted class name
        if predicted_class_name == "STYLE & BEAUTY":
            travel_image = Image.open("STYLE & BEAUTY.jpg")
            st.image(travel_image, caption="Style & Beauty", use_column_width=True)

        # Display an image based on the predicted class name
        if predicted_class_name == "TRAVEL":
            travel_image = Image.open("TRAVEL.jpg")
            st.image(travel_image, caption="Travel", use_column_width=True)

        # Display an image based on the predicted class name
        if predicted_class_name == "WEDDINGS":
            travel_image = Image.open("WEDDINGS.jpg")
            st.image(travel_image, caption="Weddings", use_column_width=True)

        # Display an image based on the predicted class name
        if predicted_class_name == "WELLNESS":
            travel_image = Image.open("WELLNESS.jpg")
            st.image(travel_image, caption="Wellness", use_column_width=True)

        # Display an image based on the predicted class name
        if predicted_class_name == "WOMEN":
            travel_image = Image.open("WOMEN.jpg")
            st.image(travel_image, caption="Women", use_column_width=True)

        # Display an image based on the predicted class name
        if predicted_class_name == "WORLD NEWS":
            travel_image = Image.open("WORLD NEWS.jpg")
            st.image(travel_image, caption="World News", use_column_width=True)


    # Total number of categories (0 to 21)
    total_categories = 22

    # Iterate through all categories
    for category in range(total_categories):
        # Check if the predicted category matches the current category
        if y_pred_new[0] == category:
            # Set the title font size to 22px and underline it using HTML formatting
            st.markdown("<h1 style='font-size: 22px; text-decoration: underline;'>Related Articles</h1>", unsafe_allow_html=True)
            
            # Load the recommendation data from "recommendation.csv" with 'latin1' encoding
            recommendation_df = pd.read_csv("recommendation.csv", encoding='latin1')
            # Filter the recommendation_df for the current category
            category_recommendations = recommendation_df[recommendation_df['Category'] == category]
            # Check if there are recommendations for the current category
            if not category_recommendations.empty:
                # Create three equal-sized columns for displaying pairs side by side
                col1, col2, col3 = st.columns(3)
                # Initialize variables to keep track of column assignments
                columns = [col1, col2, col3]
                current_column = 0
                # Initialize a variable to keep track of the maximum text height
                max_text_height = 0
                # Display each headline and short description pair in separate columns
                for index, row in category_recommendations.iterrows():
                    with columns[current_column]:
                        # Calculate the text height for the current content
                        text_height = max(len(row['Headline']), len(row['Short Description']))
                        # Calculate padding to align with the maximum text height
                        padding = " " * (max_text_height - text_height)
                        # Update the maximum text height if needed
                        max_text_height = max(max_text_height, text_height)
                        st.markdown(f"**{row['Headline']}**")
                        st.markdown(f"{row['Short Description']}")
                        st.write(padding)  # Add padding to align with the maximum text height
                        # st.write("___")  # Add a horizontal line to separate pairs
                    # Update the current column assignment (looping through 0, 1, 2)
                    current_column = (current_column + 1) % 3
            else:
                st.markdown("No recommendations available for this category.", unsafe_allow_html=True)




        # # Move "Predicted Results" to the bottom of the page within the container
        # with prediction_container:
        #     st.markdown("### Predicted Results:")
        #     st.write("Predicted Class Label:", y_pred_new[0])
        #     st.write("Predicted Class Name:", predicted_class_name)

# Handle navigation based on the selected menu option
# You can add content for each section here if needed
