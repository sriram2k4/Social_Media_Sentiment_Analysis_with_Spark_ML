import csv
import random
from datetime import datetime, timedelta

num_rows = 100000

positive_templates = [
    "I love this product!",
    "This service is excellent!",
    "I'm very happy with my experience!",
    "What an awesome experience!",
    "Absolutely great service!",
    "I highly recommend this!",
    "The quality is fantastic!",
    "I'm impressed by the results!",
    "This is the best I've ever seen!",
    "Truly outstanding performance!"
]

negative_templates = [
    "I hate this product!",
    "This service is terrible!",
    "I'm very disappointed!",
    "What a horrible experience!",
    "Absolutely awful service!",
    "I would not recommend this!",
    "The quality is poor!",
    "I'm frustrated with the results!",
    "This is the worst I've ever seen!",
    "Truly unsatisfactory performance!"
]

neutral_templates = [
    "It is okay, nothing special.",
    "I'm indifferent about this.",
    "It meets my expectations.",
    "It's average, not too good not too bad.",
    "I feel neutral about this.",
    "This is standard.",
    "No strong feelings either way.",
    "It's fine as it is.",
    "Moderate experience overall.",
    "Simply acceptable."
]

users = [f"user{i}" for i in range(1, 101)]

start_date = datetime(2021, 1, 1)
end_date = datetime(2022, 1, 1)
delta = end_date - start_date

output_file = "social_media_large_dataset.csv"

with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "timestamp", "user", "text", "sentiment"])
    
    for i in range(1, num_rows + 1):
        sentiment = random.choice(["Positive", "Negative", "Neutral"])
        
        if sentiment == "Positive":
            text = random.choice(positive_templates)
        elif sentiment == "Negative":
            text = random.choice(negative_templates)
        else:
            text = random.choice(neutral_templates)
        
        random_seconds = random.randint(0, int(delta.total_seconds()))
        timestamp = start_date + timedelta(seconds=random_seconds)
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        user = random.choice(users)
        
        writer.writerow([i, timestamp_str, user, text, sentiment])

print(f"Dataset generated: {output_file} with {num_rows} rows.")
