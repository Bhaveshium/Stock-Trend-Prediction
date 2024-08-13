<h1 align="center" id="title">Stock Trend Prediction</h1>

📄 <b>Overview</b> <br>
This project focused on predicting stock prices using a Long Short-Term Memory (LSTM) neural network. This project aims to apply machine learning techniques to forecast the closing price of a specific stock based on historical data. The repository includes data preprocessing, model building, training, and evaluation code.
  
  <img src="https://github.com/Bhaveshium/Sales-Analysis-with-Power-BI/blob/main/SuperStore%20Sales%20Dashboard.jpeg" alt="project-screenshot" width="921" height="526/">


🛠️ <b>Project Components </b> <br> 
<b>Data Source: </b>  The project utilizes stock data retrieved from Yahoo Finance (yfinance library) for a given period. <br> 
<b>Data Processing: </b>  The dataset is processed to calculate moving averages and normalize the values to prepare them for training. <br> 
<b>Model Architecture: </b>  LSTM layers with varying units (50, 60, 80, 120). <br> 
Dropout layers to prevent overfitting. <br> 
Dense layer to compile the final output. <br> 
<b>Training: </b>  The model is trained on 70% of the data, with the remaining 30% used for testing and validation. <br> 
<b>Evaluation: </b>  The model's performance is evaluated by comparing predicted stock prices against actual prices. <br> 

📊<b> Key Features </b> <br> 
<b>Moving Averages: </b>  Computation of 100-day and 200-day moving averages to smooth out the data. <br> 
<b>Data Normalization: </b>  Scaling of data using MinMaxScaler to improve model accuracy. <br> 
<b>LSTM Network: </b>  Implementation of a deep learning model with multiple LSTM layers to capture long-term dependencies in stock price data. <br> 
<b>Prediction: </b>  Generation of future stock price predictions, visualized against actual prices. <br> 

📈 <b> Results </b> <br> 
🔹The LSTM model successfully predicts stock prices with a reasonable level of accuracy. <br> 
🔹Visual comparisons between actual and predicted prices indicate the model's capability to capture general trends. <br> 

🔍<b>  Insights and Findings </b> <br> 
🔹The model captures the upward and downward trends in stock prices, although some sharp price movements may be challenging to predict accurately. <br> 
🔹Moving averages help in smoothing the price data, which improves the model's prediction stability. <br> 
  
<h2>💻 Built with</h2>

Technologies used in the project:

*   Jupyter Notebook
*   Python script
