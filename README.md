# Agency Evals

The purpose of this project is to operationalise conceptions of agency and then develop methods for testing the degree to which different LLMs enhance or diminish the agency of users depending on these criteria. We do this by identifying behaviors in LLMs that may impact human agency and then developing methods for testing the degree to which these behaviors are present in different LLMs. Each of these subdimensions are developed as a dataset of prompts which elicit the agency-influencing behavior and then a set of responses which are scored for the presence of the behavior.

## Dataset Generation Pipeline 

The dataset generation pipeline is implemented via the pipeline.py file. This file aggregates functions from the rest of the libary to generate and visualise a subdimension of choice. The subdimension of choice is passed to pipeline.py as a commandline argument. We implement the following subdimensions:

- CorrectMisinformation 
- CitesSources
- Myopia
