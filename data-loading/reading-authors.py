import redis

# 1. Connect to your Redis instance
r = redis.Redis(
    host='localhost', 
    port=6379, 
    db=0, 
    decode_responses=True # Crucial to return strings instead of bytes
)

# 2. Initialize a set to automatically track UNIQUE entries
unique_authors = set()
total_pairs_tracked = 0

print("Scanning Redis database for dataset entries...")

# 3. Use scan_iter to safely iterate over keys matching your fandom schema
# Change 'dataset:fandom:pair:*' to match your exact pattern if using another set (e.g., 'dataset:pan2023:*')
for key in r.scan_iter("dataset:fandom:pair:*"):
    total_pairs_tracked += 1
    
    # Fetch the specific metadata tracking fields from the Hash map
    # If your dataset maps individual authors, replace "fandom_1" with "author_1" etc.
    fields = r.hmget(key, ["fandom_1", "fandom_2"])
    
    # Add non-empty records to our unique set container
    if fields[0]:
        unique_authors.add(fields[0])
    if fields[1]:
        unique_authors.add(fields[1])

# 4. Print out the final dataset statistics
print("\n================ Redis Database Report ================")
print(f"Total Text Pairs Indexed:     {total_pairs_tracked}")
print(f"Total Unique Authors/Fandoms: {len(unique_authors)}")
print("=======================================================")
print("List of Unique Writers/Domains found:")
for idx, author in enumerate(sorted(unique_authors), 1):
    print(f"  {idx}. {author}")