import os
import random
import pandas as pd
from datetime import datetime, timedelta

def create_dirs():
    dirs = [
        "data/raw",
        "data/processed",
        "models/sentiment_model",
        "notebooks",
        "reports",
        "src",
        "scripts"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")

def generate_synthetic_data(num_records=2000):
    influencers = ["@fashionista_x", "@style_guru", "@vintage_vibes", "@hypebeast_daily", "@minimal_chic"]
    trends = ["Y2K", "streetwear", "quiet luxury", "coquette", "athleisure", "oversized clothing", "denim", "cargo pants", "ballet flats", "monochrome", "sustainable fashion", "vintage"]
    
    # Templates for captions
    caption_templates = [
        "Obsessed with this {trend} look! #fashion #{trend_no_space}",
        "My take on the {trend} trend. What do you guys think?",
        "Loving the {trend} aesthetic lately. So comfy yet stylish.",
        "Can't get enough of {trend}. It's taking over my wardrobe!",
        "Classic {trend} vibes for today's outfit. #OOTD",
    ]
    
    # Templates for comments (positive, negative, neutral)
    comment_templates = [
        ("I absolutely love this {trend} style!", "positive"),
        ("This {trend} look is everything!", "positive"),
        ("Not a fan of {trend}, it looks weird to me.", "negative"),
        ("I think {trend} is overrated tbh.", "negative"),
        ("It's okay, {trend} is just another passing fad.", "neutral"),
        ("Where did you get that {trend} piece?", "neutral"),
        ("So stunning! {trend} fits you perfectly.", "positive"),
        ("I don't get the hype around {trend}.", "negative")
    ]

    
    start_date = datetime(2023, 1, 1)
    
    data = []
    
    for i in range(num_records):
        influencer = random.choice(influencers)
        # Random date within the last year
        date = start_date + timedelta(days=random.randint(0, 365))
        trend = random.choice(trends)
        trend_no_space = trend.replace(" ", "")
        
        is_caption = random.choice([True, False])
        
        if is_caption:
            text = random.choice(caption_templates).format(trend=trend, trend_no_space=trend_no_space)
            # Most captions are positive or neutral
            sentiment = random.choice(["positive", "positive", "neutral"])
        else:
            comment, sentiment = random.choice(comment_templates)
            text = comment.format(trend=trend)
            
        data.append({
            "post_id": f"P{i:05d}",
            "influencer": influencer,
            "date": date.strftime("%Y-%m-%d"),
            "text": text,
            "is_caption": is_caption,
            "hashtags": f"#{trend_no_space}" if is_caption else "",
            "likes": random.randint(10, 5000) if is_caption else random.randint(0, 100),
            "true_trend": trend,
            "sentiment_label": sentiment
        })
        
    df = pd.DataFrame(data)
    # Sort by date
    df = df.sort_values(by="date")
    
    output_path = "data/raw/synthetic_fashion_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {num_records} synthetic records and saved to {output_path}")

if __name__ == "__main__":
    create_dirs()
    generate_synthetic_data(2500)
