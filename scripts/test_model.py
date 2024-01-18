import pandas as pd
import spacy
from spacy.tokens import Doc, Span

import rel_model

nlp = spacy.load("en_core_web_sm")

text = """Le capital social est fixé à la somme de 40 000 Euros ( quarante mille euros ) divisé en 400
( quatre cents ) parts de 100 ( cent ) Euros chacune."""
doc = nlp(text)

entity_dict = {
    (9, 11): "CARDINAL",
    (19, 20): "CARDINAL",
    (27, 28): "MONEY",
}

# Initialize entity spans
entities = []
for (start, end), label in entity_dict.items():
    span = Span(doc, start, end, label=label)
    entities.append(span)

# Set the entities on the Doc
doc.ents = entities

print(entities)

from rel_pipe import make_relation_extractor

# Load the RE model (assuming you've already loaded it)
nlp2 = spacy.load("training/model-last")

for name, proc in nlp2.pipeline:
          doc = proc(doc)

# doc._.rel contains the extracted relations
print('doc._.rel.items()',doc._.rel.items())

print('doc.sents',doc.sents)

best_relation_list = []

for value, rel_dict in doc._.rel.items():
    for e in doc.ents:
        for b in doc.ents:
            if e.start == value[0] and b.start == value[1] and e != b:
                max_confidence = max(rel_dict.values(), default=-1)
                best_relation = next((relation for relation, confidence in rel_dict.items() if confidence == max_confidence), None)

                if best_relation is not None:
                    # Add the result to the list
                    result_dict = {
                        "entity1": e.text,
                        "relation": best_relation,
                        "entity2": b.text,
                        "confidence": max_confidence
                    }
                    best_relation_list.append(result_dict)

dataframes_dict = {}

for entry in best_relation_list:
    relation = entry['relation']

    if relation not in dataframes_dict:
        dataframes_dict[relation] = pd.DataFrame(columns=["entity1", "relation", "entity2", "confidence"])

    dataframes_dict[relation] = pd.concat([dataframes_dict[relation], pd.DataFrame([entry])], ignore_index=True)

html_file_path = "report.html"

with open(html_file_path, "w") as html_file:
    html_file.write("<html><head><title>Report</title></head><body>")
    for relation, df in dataframes_dict.items():
        html_file.write(f"<h2>Relation '{relation}':</h2>")
        html_file.write(df.to_html(index=False))

    html_file.write("</body></html>")

print(f"Report has been saved to: {html_file_path}")