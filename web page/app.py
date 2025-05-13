from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import io
import base64
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    days_to_predict = int(request.form['days'])

    if file and allowed_file(file.filename):
        filename = 'uploaded_file.csv'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Read CSV data
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        df.set_index('Date', inplace=True)

        # Data preprocessing
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df['Price'].values.reshape(-1, 1))

        sequence_length = 60

        def create_dataset(data, sequence_length):
            x, y = [], []
            for i in range(len(data) - sequence_length):
                x.append(data[i:i + sequence_length])
                y.append(data[i + sequence_length])
            return np.array(x), np.array(y)

        x, y = create_dataset(scaled_data, sequence_length)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])

        # Linear Regression model
        model = LinearRegression()
        model.fit(x_train, y_train)

        # Predict future prices
        last_sequence = scaled_data[-sequence_length:].reshape(1, -1)
        predicted_prices = []

        for _ in range(days_to_predict):
            next_day_prediction = model.predict(last_sequence)
            next_day_price = scaler.inverse_transform(next_day_prediction.reshape(-1, 1))[0][0]
            predicted_prices.append(next_day_price)

            # Update the last_sequence with the predicted price for the next iteration
            last_sequence = np.append(last_sequence[:, 1:], next_day_prediction).reshape(1, -1)

        # Calculate accuracy metrics (based on test set)
        mse = mean_squared_error(y_test, model.predict(x_test))
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, model.predict(x_test))
        r2 = r2_score(y_test, model.predict(x_test))

        # Plot actual and predicted results
        plt.figure(figsize=(15, 5))
        plt.plot(scaler.inverse_transform(scaled_data), label='Actual Data', color='blue')
        plt.axvline(x=len(scaled_data) - sequence_length, color='gray', linestyle='--', label='Prediction Start')
        plt.scatter(np.arange(len(scaled_data), len(scaled_data) + days_to_predict), predicted_prices, color='red', label='Predicted Future Prices')
        plt.title(f'Price Prediction for Next {days_to_predict} Days')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()

        # Save plot to a BytesIO object and encode to base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()

        return render_template('result.html', mse=mse, rmse=rmse, mae=mae, r2=r2, img_data=img_base64, predicted_prices=predicted_prices, days_to_predict=days_to_predict)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
