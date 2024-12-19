import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Function for training linear regression model
def train_model(df):
    X = df[['size_sqft']]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

# Streamlit User Interface for the Application
def main():
    st.title('🏠 House Pricing Predictor')
    st.write('Upload a dataset with `size_sqft` and `price` columns to train the model.')

    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=['csv'])

    if uploaded_file is not None:
        # Load dataset
        df = pd.read_csv(uploaded_file)

        # Ensure required columns are present
        if 'size_sqft' not in df.columns or 'price' not in df.columns:
            st.error("The dataset must contain 'size_sqft' and 'price' columns.")
            return

        st.write("### Dataset Preview")
        st.dataframe(df.head())

        # Train the model
        model, X_test, y_test = train_model(df)
        
        # Test data selection
        st.write("### Select a sample from the test set to predict")
        sample_idx = st.selectbox("Select test sample index", X_test.index)

        if st.button('Predict Price'):
            # Perform prediction
            selected_size = X_test.loc[sample_idx, 'size_sqft']
            prediction = model.predict([[selected_size]])
            actual_price = y_test.loc[sample_idx]

            st.success(f"Predicted price: ${prediction[0]:,.2f}")
            st.info(f"Actual price: ${actual_price:,.2f}")

            # Visualization
            fig = px.scatter(df, x='size_sqft', y='price', title='Size vs Price Relationship')
            fig.add_scatter(x=[selected_size], y=[prediction[0]], mode='markers', 
                            marker=dict(size=15, color='red'), name='Prediction')
            st.plotly_chart(fig)

if __name__ == '__main__':
    main()
