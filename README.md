# setsolver-api
The API for the the SET solver. 
Idea: 
- Gets an image of (not overlapping) set cards and detects valid sets.
- API implemented with Flask and model trained with pytorch (see setsolver-card-classifier project)

## run locally
```
export FLASK_APP=setsolverapi
flask run --host=0.0.0.0
```