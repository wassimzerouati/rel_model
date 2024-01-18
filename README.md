# Relation Extraction Model Update Instructions

Follow these steps to update and retrain the relation extraction model:

```bash
# Clone the Repository
git clone https://github.com/wassimzerouati/rel_model.git
cd rel_model

# Update the Annotation Data
# (Update the relation_training.txt file in the 'data' folder with new annotations.)

# Run Binary Converter Script
python scripts/binary_converter.py

# Retrain the Model
spacy project run train_cpu

# Copy the Model-last Folder
# (Copy the 'model-last' folder to the corresponding path in the extension in Papai. Ensure the extension in Papai is configured to use the updated model.)
