import json

# Assume the original data is stored in data.json
with open('./peel-banana/dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Filter out entries in train_ids and val_ids that start with "right"
for key in ('train_ids', 'val_ids'):
    data[key] = [item for item in data[key] if not item.startswith('right')]

# Keep the ids field unchanged (including both left and right), and update count to match the length of ids
data['count'] = len(data['ids'])
# Keep num_exemplars unchanged

# Write the result back to file
with open('./peel-banana-single/dataset.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Filtering complete: kept {data['count']} ids, {len(data['train_ids'])} train_ids, {len(data['val_ids'])} val_ids.")
