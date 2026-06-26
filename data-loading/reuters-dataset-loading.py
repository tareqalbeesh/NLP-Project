import os
import redis

r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Point this to your local folder containing the author subdirectories
C50_DIRECTORY = "../dataset/reuter+50+50/C50train" 

loaded_count = 0

for author_folder in os.listdir(C50_DIRECTORY):
    author_path = os.path.join(C50_DIRECTORY, author_folder)
    
    # Verify that it is an author directory
    if os.path.isdir(author_path):
        doc_counter = 0
        
        for txt_file in os.listdir(author_path):
            if txt_file.endswith(".txt"):
                file_path = os.path.join(author_path, txt_file)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read().strip()
                    
                if len(raw_text) < 50:
                    continue
                    
                redis_key = f"dataset:reuters5050:author:{author_folder}:doc:{doc_counter:03d}"
                
                payload = {
                    "author": author_folder,
                    "text": raw_text
                }
                
                r.hset(redis_key, mapping=payload)
                doc_counter += 1
                loaded_count += 1

print(f"Direct migration complete. Stored {loaded_count} files for Reuters 50-50.")