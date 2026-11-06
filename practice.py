responses = [
       {"id": 1, "text": "Bus fares are too expensive for families on low incomes.", "themes": ["fares"]},
       {"id": 2, "text": "Services in my village were cut and now I can't get to work.", "themes": ["cuts"]},
       {"id": 3, "text": "Buses are often late, so I've started driving instead.", "themes": ["reliability"]},
       {"id": 4, "text": "Cheaper fares would help, but reliability matters more.", "themes": ["fares", "reliability"]},
       {"id": 5, "text": "There should be more evening services for shift workers.", "themes": ["evening"]},
       {"id": 6, "text": "I can't use the bus because the stop isn't accessible.", "themes": ["accessibility"]}]

# Loop through and print each id with the first 30 characters of its text.
for i, response in enumerate(responses):
    print (i, response)

for i, response in enumerate(responses):
    print(f"Response {response['id']}: {response['text'][:30]}")

fares_response_ids = []
count = 0

#Build a list of the ids whose themes include "fares". Expected: [1, 4].
#Count how many responses have more than one theme. Expected: 1.
for i, response in enumerate(responses):
    print(f"{response['id']}: {response['text'][:30]}")
    if "fares" in response["themes"]:
        print("  This response mentions fares.")
        fares_response_ids.append(response['id'])
        count += 1
    print(f'list of fares responses: {fares_response_ids}')
print(f"Number of responses mentioning fares: {count}")

a = sorted(responses, key=lambda x: (len(x["themes"]), x["id"]))
print(f'sorted by themes and id: {a}')

#Count how many responses have more than one theme. Expected: 1.
for i, response in enumerate(responses):
    if len(response['themes']) >1:
        print(f"Response {response['id']} has more than one theme: {response['themes']}")

#Sort the texts by length

texts_by_length = sorted([r["text"] for r in responses], key=len)
print(f'texts_by_length: {texts_by_length}')

total_themeCount = 0

from collections import Counter

CounterThemes = Counter()
for i, response in enumerate(responses):
    CounterThemes.update(response["themes"])
    total_themeCount += len(response["themes"])

print(f'CounterThemes: {CounterThemes}')
print(f'total_themeCount: {total_themeCount}')