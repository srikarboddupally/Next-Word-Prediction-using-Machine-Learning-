import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics

data = {'text': ["This is a sample sentence",
                 "Next word prediction is an interesting task",
                 "Machine learning can be used for various applications",
                 "Python is a popular programming language","My name is Srikar"]}

df = pd.DataFrame(data)

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)


model = make_pipeline(CountVectorizer(), MultinomialNB())

model.fit(train_data['text'], train_data.index)

predictions = model.predict(test_data['text'])

accuracy = metrics.accuracy_score(test_data.index, predictions)
print(f"Accuracy: {accuracy:.2%}")

def predict_next_word(text):
    predicted_index = model.predict([text])[0]
    predicted_word = df.loc[predicted_index, 'text'].split()[-1]
    return predicted_word

input_text = input("Enter a sentence : ")
predicted_next_word = predict_next_word(input_text)
print(f"The predicted next word after '{input_text}' is: {predicted_next_word}")
