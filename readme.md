# 📧 Email/SMS Spam Classifier 

This is a machine learning project that classifies emails and SMS messages as spam or not spam using Natural Language Processing (NLP) where I have used Navie Based ( Multinomial Naive Bayes ) which assumes that all the word in a sentence does not have correlation, this may not be true in practical situations but when I tried this model it gave some really good scores of 97.10 % accuracy and a precison of 100 % after using several EDA ( Exploratory Data Analysis ) methods and feature engineering methods I broke the dataset into 80 % training and 20 % tesitng and then I was able to get usefull features to feed the model.

## Second Enhancement
Further I used docker to containarize my model so that anyone can use docker image and build my project locally and don't have to install all the dependencies. I also hosted this model on Huggingface which provided me smooth integration of model backend with streamlit frontend.

## 👨‍💻 Author
**Shrish Mishra**

## 🎯 Project Overview

This project implements a spam detection system using multiple machine learning algorithms. The system processes text messages, transforms them using TF-IDF vectorization, and classifies them as spam or legitimate (ham) messages. 

## 📚 Dataset

- **Source**: SMS Spam Collection Dataset 
- **Total Messages**: 5,572 messages 
- **After Preprocessing**: 5,169 messages (after removing 403 duplicates) 
- **Distribution**:
  - Ham (Not Spam): 4,516 messages (87.37%) 
  - Spam: 653 messages (12.63%) 

## ⚙️ Features

The project analyzes the following text features: 
- Number of characters 
- Number of words 
- Number of sentences 
- Transformed text (after preprocessing) 

## 🔧 Text Preprocessing Pipeline

The [`transform_text`](Spam_detection.ipynb) function in [Spam_detection.ipynb](Spam_detection.ipynb) and [app.py](app.py) performs the following steps: 

1. **Lowercase Conversion**: Converts all text to lowercase 
2. **Tokenization**: Breaks text into individual words using `nltk.word_tokenize()` 
3. **Alphanumeric Filtering**: Removes special characters and keeps only alphanumeric tokens 
4. **Stop Words Removal**: Removes common English stop words and punctuation 
5. **Stemming**: Reduces words to their root form using Porter Stemmer 

## 🔢 Vectorization

**TF-IDF Vectorizer** (Term Frequency-Inverse Document Frequency) 
- Maximum features: 3,000
- Converts text into numerical vectors 
- Saved as: `vectorizer.pkl` 

## 🤖 Machine Learning Models

The project evaluates 10 different classification algorithms: 

### 📊 Model Performance Comparison

| Algorithm | Accuracy | Precision |
|-----------|----------|-----------|
| **Multinomial Naive Bayes (NB)**  | **97.10%** | **100.00%** |
| K-Nearest Neighbors (KN) | 90.52% | 100.00% |
| Random Forest (RF) | 97.58% | 98.29% |
| Support Vector Classifier (SVC) | 97.58% | 97.48% |
| Extra Trees Classifier (ETC) | 97.49% | 97.46% |
| Logistic Regression (LR) | 95.84% | 97.03% |
| Gradient Boosting (GBDT) | 94.68% | 91.92% |
| Bagging Classifier (BgC) | 95.84% | 86.82% |
| AdaBoost | 92.46% | 84.88% |
| Decision Tree (DT) | 92.75% | 81.19% |

### 🏆 Best Performing Model

**Multinomial Naive Bayes** was selected as the final model due to: 
- High accuracy: 97.10% 
- Perfect precision: 100.00% 
- No false positives (0 legitimate messages classified as spam) 
- Confusion Matrix: 
  ```
  [[896   0]
   [ 30 108]]
  ```
- Model saved as: `model.pkl` 💾

## 📁 Project Structure

```
spam_detection_model/
├── app.py                      # Streamlit web application 
├── Spam_detection.ipynb        # Jupyter notebook with full analysis 
├── spam.csv                    # Dataset 
├── model.pkl                   # Trained Multinomial Naive Bayes model 
└── vectorizer.pkl             # TF-IDF vectorizer 
```

## 🛠️ Technologies Used

- **Python 3.9.6** 🐍
- **Libraries**:
  - pandas 
  - numpy 
  - nltk 
  - scikit-learn 
  - matplotlib 
  - seaborn 
  - wordcloud 
  - streamlit 
  - pickle 

## 💻 Installation

1. Clone the repository 
2. Install required packages: 
```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn wordcloud streamlit
```

3. Download NLTK data: ⬇️
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
```

## 🚀 Usage

### 🌐 Running the Web Application

```bash
streamlit run app.py
```

The application provides a simple interface where you can: 
1. Enter an email or SMS message 
2. Click "Predict" 
3. See if the message is classified as "Spam" or "Not Spam" 

### 📓 Using the Notebook

Open [`Spam_detection.ipynb`](Spam_detection.ipynb) in Jupyter Notebook to: 
- Explore the complete data analysis 
- View visualizations (word clouds, histograms, correlation heatmaps) 
- Train and evaluate different models 
- Modify and experiment with the code 

## 🎓 Model Training Process

1. **Data Loading**: Read spam.csv with ISO-8859-1 encoding 
2. **Data Cleaning**: 
   - Remove unnecessary columns 
   - Rename columns to 'target' and 'text' 
   - Encode labels (ham=0, spam=1) 
   - Remove 403 duplicate messages 
3. **Feature Engineering**: Extract character count, word count, and sentence count 
4. **Text Preprocessing**: Apply the [`transform_text`](Spam_detection.ipynb) function 
5. **Vectorization**: Convert text to TF-IDF features (3000 features) 
6. **Train-Test Split**: 80% training, 20% testing (random_state=2) 
7. **Model Training**: Train 10 different classifiers 
8. **Evaluation**: Compare accuracy and precision scores 
9. **Model Selection**: Choose Multinomial Naive Bayes as the final model 

## 💡 Key Insights

- Spam messages are significantly longer than ham messages 📏
- Average characters: 
  - Ham: 70.46 
  - Spam: 137.89 
- Average words: 
  - Ham: 17.12 
  - Spam: 27.67 
- Most frequent words in spam: "call", "free", "txt", "claim", "prize" 

## 🌐 Deployment

Run locally with:
```bash
streamlit run app.py
```

### Docker (Build and Run)
You can run the app inside Docker — this is useful for consistent environments and for deploying to servers or CI.

Build the Docker image locally (from the repository root):
```bash
docker build -t spam-detection-app:latest .
```

Run the container and map Streamlit's port 8501 to your host:
```bash
docker run --rm -p 8501:8501 -v "$PWD":/app spam-detection-app:latest
```

Or use docker-compose (recommended for development):
```bash
docker compose up --build
```

Example: run the container and pass a private HF token (if needed):
```bash
docker run --rm -p 8501:8501 -e HF_TOKEN="$HF_TOKEN" -v "$PWD":/app spam-detection-app:latest
```


## 🚀 Future Improvements

- Add more sophisticated preprocessing techniques 
- Implement deep learning models (LSTM, BERT) 
- Add multilingual support 
- Enhance the web interface with more features 
- Add confidence scores and probability display 
- Implement feedback mechanism for model improvement 

## 📄 License

This project is open source and available for educational purposes. 📖

## 📧 Contact

For questions or suggestions, please contact Shrish Mishra. 💬
## Future Improvements

- Add more sophisticated preprocessing techniques
- Implement deep learning models (LSTM, BERT)
- Add multilingual support
- Enhance the web interface with more features
- Deploy to cloud platform (Heroku, AWS, etc.)

## Contact

For questions or suggestions, please contact at shrish409@gmail.com
