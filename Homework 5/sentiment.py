from transformers import pipeline
sentiment_analysis = pipeline("sentiment-analysis")
pos_text = "I enjoy studying computational algorithms."
neg_text = "I dislike sleeping late everyday."
result = sentiment_analysis(pos_text)[0]
print("Label:", result['label'])
print("Confidence Score:", result['score'])
print()
result = sentiment_analysis(neg_text)[0]
print("Label:", result['label'])
print("Confidence Score:", result['score'])