## Project Overview

This project performs **multilingual sentiment classification** in:
- English
- Spanish
- French  

The system takes a sentence or short paragraph as input and returns:
- A sentiment label (Positive, Neutral, or Negative)
- A confidence score

--------------------------------------------------------------------------

Running the Application:

Predict sentiment
python src/main.py predict --text "This movie was amazing" --lang en

Explain prediction
python src/main.py explain --text "Esta película fue terrible" --lang es

Evaluate dataset
python src/main.py evaluate --file data/sample/en_sample.csv