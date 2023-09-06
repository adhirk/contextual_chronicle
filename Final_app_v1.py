import streamlit as st
import pickle
from PIL import Image
import datetime  # Import the datetime module

# Load the saved model, vectorizer, and label encoder
model_filename = 'logistic_regression_model.pkl'
vectorizer_filename = 'tfidf_vectorizer_lem.pkl'
label_encoder_filename = 'label_encoder.pkl'

with open(model_filename, 'rb') as model_file:
    loaded_classifier = pickle.load(model_file)

with open(vectorizer_filename, 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

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
        <span style="border-top: 2px solid #000; width: 60%; margin-top: 10px; display: inline-block;"></span><br>
        <span style="height: 20px; display: block;"></span>  <!-- Add 20px space -->
    </div>
    ''',
    unsafe_allow_html=True
)

# Create the fixed box for menu items with options, increased spacing, and space below
box_style = """
    border: 1px solid #ccc;
    padding: 10px;
    border-radius: 5px;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 50px; /* Add 50px space below the menu box */
    display: flex;
    flex-direction: column; /* Display menu items vertically */
    align-items: center; /* Center align menu items horizontally */
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


# Add dynamic date and time below the menu box in the desired format
current_datetime = datetime.datetime.now().strftime("%b %d, %Y, %I:%M %p")
formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.write(current_datetime)

# Sidebar for user input
st.sidebar.title("User Input")

input_headline = st.sidebar.text_input("Enter the headline:")
input_article = st.sidebar.text_area("Enter the article text:")

if st.sidebar.button("Predict"):
    input_sentence = input_headline + ' ' + input_article
    X_new_data_transformed = loaded_vectorizer.transform([input_sentence])
    y_pred_new = loaded_classifier.predict(X_new_data_transformed)
    predicted_class_name = loaded_label_encoder.inverse_transform(y_pred_new)[0]

    # Create two equal-sized columns
    col1, col2 = st.columns(2)

    # Display predictions in the left column
    with col1:
        st.markdown(
            f'<div style="font-family: Abril Fatface; font-size: 24px;">{input_headline}</div>',
            unsafe_allow_html=True
        )
        st.write(input_article)  # Display the article text without "Headline" text

    # Create a container to hold the "Predicted Results" section
    prediction_container = st.container()

    # Display predicted images in the right column with adjusted size
    with col2:
        # Display an image based on the predicted class name
        if predicted_class_name == "TRAVEL":
            travel_image = Image.open("TRAVEL.jpg")
            st.image(travel_image, caption="Travel Image", use_column_width=True)

        # Display an image based on the predicted class name
        if predicted_class_name == "SPORTS":
            sports_image = Image.open("SPORTS.jpg")
            st.image(sports_image, caption="Sports Image", use_column_width=True)

        # Move "Predicted Results" to the bottom of the page within the container
        with prediction_container:
            st.markdown("### Predicted Results:")
            st.write("Predicted Class Label:", y_pred_new[0])
            st.write("Predicted Class Name:", predicted_class_name)

# Handle navigation based on the selected menu option
# You can add content for each section here if needed
