# Step 1: Import necessary libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Step 2: Load and preprocess the data
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

def preprocess_data(data):
    # Fill missing values and encode categorical features
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)
    
    # Encoding categorical columns
    label_encoder = LabelEncoder()
    data['Sex'] = label_encoder.fit_transform(data['Sex'])
    data['Embarked'] = label_encoder.fit_transform(data['Embarked'])
    
    return data

# Step 3: Sidebar for user input
st.sidebar.title("Titanic Survival Prediction")

# Step 4: File uploader for CSV input
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

# Step 5: Display data after upload
if uploaded_file is not None:
    data = load_data(uploaded_file)
    data = preprocess_data(data)
    
    # Step 6: Allow the user to display the dataset
    if st.sidebar.checkbox("Show Dataset"):
        st.subheader("Titanic Dataset")
        st.write(data)
    
    # Sidebar feature selection
    st.sidebar.subheader("Select Features")
    selected_features = st.sidebar.multiselect(
        "Select features to include in the model", 
        options=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], 
        default=['Pclass', 'Sex', 'Age', 'Fare']
    )

    # Add a slider for selecting the test size
    test_size = st.sidebar.slider(
        "Select test size", 
        min_value=0.1, max_value=0.5, step=0.05, value=0.2
    )

    # Add a dropdown for selecting the classifier
    classifier_name = st.sidebar.selectbox(
        "Select Classifier", 
        ("Random Forest", "Decision Tree")
    )

    # Add a radio button for the output metric
    output_metric = st.sidebar.radio(
        "Choose a performance metric", 
        ('Accuracy', 'Confusion Matrix', 'Classification Report')
    )

    # Step 7: Add a submit button
    if st.sidebar.button("Train Model"):
        st.subheader("Training Results")

        # Split data into training and testing
        X = data[selected_features]
        y = data['Survived']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Build and train the selected model
        if classifier_name == 'Random Forest':
            model = RandomForestClassifier(random_state=42)
        else:
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display results based on user-selected metric
        if output_metric == 'Accuracy':
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"**Accuracy:** {accuracy:.2f}")

        elif output_metric == 'Confusion Matrix':
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)

        elif output_metric == 'Classification Report':
            report = classification_report(y_test, y_pred, output_dict=True)
            st.write(pd.DataFrame(report).transpose())

        # Step 8: Display y_test and y_pred in a widened table
        st.subheader("Predicted vs Actual Results")
        results_df = pd.DataFrame({"Actual ": y_test, "Predicted": y_pred})

        # Customize the display to show wider columns
    # Use raw HTML to create a scrollable, fixed-width table
        st.markdown(
            """
            <style>
            .scrollable-table {
                max-width: 800px;
                max-height: 300px;
                overflow: auto;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                padding: 10px;
                border: 1px solid black;
                text-align: center;
            }
            </style>
            """, 
            unsafe_allow_html=True
        )

        # Display the table with scrollable class
        st.markdown(f"""
        <div class="scrollable-table">
        {results_df.to_html(index=False)}
        </div>
        """, unsafe_allow_html=True)

else:
    st.write("Please upload a CSV file to start the analysis.")
