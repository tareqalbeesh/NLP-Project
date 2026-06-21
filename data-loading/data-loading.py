import redis
import nltk
import os
# 1. Choose a clean, non-sandboxed directory for NLTK data
custom_nltk_path = os.path.join(os.path.expanduser("~"), "nltk_data")

# 2. Tell NLTK to look ONLY in this specific safe directory
nltk.data.path = [custom_nltk_path]

# 3. Explicitly download the dataset into your custom safe folder
print("Downloading Gutenberg dataset to a secure path...")
nltk.download('gutenberg', download_dir=custom_nltk_path)

from nltk.corpus import gutenberg


# Ensure NLTK dataset is downloaded locally
nltk.download('gutenberg')

# Connect to your local or remote Redis instance
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def load_full_text_to_redis(file_id, author_label, target_words=600):
    print(f"Extracting texts from {file_id}...")
    
    # Fetch raw words from NLTK corpus
    words = gutenberg.words(file_id)
    total_words = len(words)
    chunk_counter = 0
    
    # Slice the entire book into contiguous chunks of ~600 words
    for i in range(0, total_words, target_words):
        chunk_words = words[i:i + target_words]
        
        # Reconstruct the raw textual body string
        full_text_content = " ".join(chunk_words)
        
        # Clean up punctuation formatting artifacts from joining tokens
        full_text_content = full_text_content.replace(" .", ".").replace(" ,", ",").replace(" ;", ";")
        full_text_content = full_text_content.replace(" ?", "?").replace(" !", "!").replace(" ' ", "'")
        
        # Validation: Ignore empty strings or small trailing fragments under 50 characters
        if len(full_text_content.strip()) < 50:
            continue
            
        # Define a predictable, highly visible Redis Key layout
        redis_key = f"dataset:gutenberg:author:{author_label}:text:{chunk_counter:04d}"
        
        # Construct the payload dictionary
        # The key 'text' contains the literal, raw text body string—not a URL or path.
        payload = {
            "author": author_label,
            "text": full_text_content
        }
        
        # Save the full payload directly into the Redis database memory
        r.hset(redis_key, mapping=payload)
        chunk_counter += 1

    print(f"Successfully saved {chunk_counter} complete text blocks for '{author_label}' directly into Redis.\n")

# ---- Run Migration ----
load_full_text_to_redis('austen-emma.txt', 'jane_austen')
load_full_text_to_redis('shakespeare-hamlet.txt', 'william_shakespeare')
load_full_text_to_redis('melville-moby_dick.txt', 'herman_melville')