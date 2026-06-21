import json
import redis

# 1. Connect to your local Redis instance
r = redis.Redis(
    host='localhost', 
    port=6379, 
    db=0, 
    decode_responses=True # Converts binary data automatically to clean Python strings
)

# Replace this with the actual path to your downloaded file
JSONL_FILE_PATH = "..\dataset\pan20-authorship-verification-test.jsonl"

print("Starting Fandom Dataset Migration to Redis...")
loaded_count = 0
skipped_count = 0

with open(JSONL_FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        # Parse the individual JSON row
        try:
            row_data = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
            
        pair_id = row_data['id']
        fandoms_list = row_data.get('fandoms', [])
        
        # Extract Text 1 and Text 2 from the 'pair' array
        text_1 = row_data['pair'][0]
        text_2 = row_data['pair'][1]
        
        # Enforce your custom n8n code node constraints:
        # Both texts must be at least 50 characters to analyze safely without failing
        if len(text_1.strip()) < 50 or len(text_2.strip()) < 50:
            skipped_count += 1
            continue
            
        # Formulate a structured Redis key pattern
        redis_key = f"dataset:fandom:pair:{pair_id}"
        
        # Build the flat map payload to save in the Redis Hash
        # Since this particular dataset doesn't explicitly state a ground truth boolean,
        # we store the fandom origins so your agent can evaluate cross-domain changes.
        payload = {
            "pair_id": pair_id,
            "fandom_1": fandoms_list[0] if len(fandoms_list) > 0 else "Unknown",
            "fandom_2": fandoms_list[1] if len(fandoms_list) > 1 else "Unknown",
            "text_1": text_1,  # Storing the entire raw text string directly in Redis memory
            "text_2": text_2   # Storing the entire raw text string directly in Redis memory
        }
        
        # Save the full payload directly into the Redis database memory
        r.hset(redis_key, mapping=payload)
        loaded_count += 1
        
        if loaded_count % 100 == 0:
            print(f"Loaded {loaded_count} text pairs into Redis...")

print(f"\nMigration Complete!")
print(f"Successfully loaded: {loaded_count} pairs.")
print(f"Skipped due to length rule (<50 chars): {skipped_count} pairs.")