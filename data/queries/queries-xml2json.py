import xml.etree.ElementTree as et
import json

topics = et.parse("./data/queries/topics-rnd5.xml").getroot()

queries = []

for topic in topics:
    current_query = {
        "number" : topic.attrib["number"]
    }
    for query in topic:
        current_query[query.tag] = query.text

    queries.append(current_query)


with open("./data/queries/queries.json", "w+") as queryF:
    json.dump(queries, queryF)