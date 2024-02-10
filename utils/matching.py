import pandas as pd
import numpy as np
import Levenshtein
from utils.metrics import jaccard_similarity

import logging

def match(row, df, weights):

    # set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    # define some features to use for matching
    last_name = row["last"]
    if not pd.isna(last_name) and not isinstance(last_name, float):
        last_name = str(last_name.lower().strip("."))
    first_name = row["first"]
    if not pd.isna(first_name) and not isinstance(first_name, float):
        first_name = str(first_name.lower().strip("."))
    middle_name = row["middle"]
    if not pd.isna(middle_name) and not isinstance(middle_name, float):
        middle_name = str(middle_name.lower().strip("."))
    year = row["year"]
    if not pd.isna(year):
        year = int(year)

    def _match_name(source_first, source_middle, source_last, source_year, df_row, weight):

        # get features from the dataframe
        # name components
        target_first = df_row["first"]
        if not pd.isna(target_first) and not isinstance(target_first, float):
            target_first = str(target_first.lower().strip("."))
        target_middle = df_row["middle"]
        if not pd.isna(target_middle) and not isinstance(target_middle, float):
            target_middle = str(target_middle.lower().strip("."))
        target_last = df_row["last"]
        if not pd.isna(target_last) and not isinstance(target_last, float):
            target_last = str(target_last.lower().strip("."))
        
        # birth and death years
        target_birth_year = df_row["birth_year"]
        if not pd.isna(target_birth_year):
            target_birth_year = int(target_birth_year)
        target_death_year = df_row["death_year"]
        if not pd.isna(target_death_year):
            target_death_year = int(target_death_year)

        # floruit years
        target_floruit_years = df_row["floruit_years"]
        if not pd.isna(target_floruit_years):
            if "-" in target_floruit_years:
                target_floruit_years = target_floruit_years.split("-")
                target_floruit_years = [int(x) for x in target_floruit_years]
            else:
                target_floruit_years = [int(target_floruit_years)-50, int(target_floruit_years)+50]
        else:
            target_floruit_years = [np.nan, np.nan]

        # initialize the component scores
        results = {
            "first_lev": 0,
            "first_jac": 0,
            "first_init": 0,
            "middle_lev": 0,
            "middle_jac": 0,
            "middle_init": 0,
            "last_lev": 0,
            "last_jac": 0,
            "last_init": 0,
            "year": 0,
            "score": 0
        }
        
        # skip if the target name is missing
        if ((pd.isna(target_first) and pd.isna(target_last) and pd.isna(target_middle)) or 
            (isinstance(target_first, float) and isinstance(target_last, float) and isinstance(target_middle, float)) or 
            (target_first == "nan" and target_last == "nan" and target_middle == "nan") or 
            (target_first == "NaN" and target_last == "NaN" and target_middle == "NaN")) or \
            (target_first == "" and target_last == "" and target_middle == ""):
            return results
        
        # skip if the source name is missing
        if ((pd.isna(source_first) and pd.isna(source_last) and pd.isna(source_middle)) or
            (isinstance(source_first, float) and isinstance(source_last, float) and isinstance(source_middle, float)) or
            (source_first == "nan" and source_last == "nan" and source_middle == "nan") or
            (source_first == "NaN" and source_last == "NaN" and source_middle == "NaN")) or \
            (source_first == "" and source_last == "" and source_middle == ""):
            return results
        
        if ((pd.isna(target_last) or isinstance(target_last, float) or pd.isna(last_name) or isinstance(last_name, float)) or 
            (target_last == "nan" or last_name == "nan") or
            (target_last == "NaN" or last_name == "NaN")) or \
            (target_last == "" or last_name == ""):
            last_lev = 0
            last_jac = 0
            last_init = 0
        else:
            last_lev = Levenshtein.distance(last_name, target_last)
            last_lev = 1 - (last_lev / max(len(last_name), len(target_last))) if max(len(last_name), len(target_last)) > 0 else 0
            last_jac = jaccard_similarity(last_name, target_last)
            last_init = 1 if last_name[0] == target_last[0] else 0
        results["last_lev"] = last_lev
        results["last_jac"] = last_jac
        results["last_init"] = last_init

        if ((pd.isna(target_first) or isinstance(target_first, float) or pd.isna(first_name) or isinstance(first_name, float)) or
            (target_first == "nan" or first_name == "nan") or
            (target_first == "NaN" or first_name == "NaN")) or \
            (target_first == "" or first_name == ""):
            first_lev = 0
            first_jac = 0
            first_init = 0
        else:
            first_lev = Levenshtein.distance(first_name, target_first)
            first_lev = 1 - (first_lev / max(len(first_name), len(target_first))) if max(len(first_name), len(target_first)) > 0 else 0
            first_jac = jaccard_similarity(first_name, target_first)
            first_init = 1 if first_name[0] == target_first[0] else 0
        results["first_lev"] = first_lev
        results["first_jac"] = first_jac
        results["first_init"] = first_init

        if ((pd.isna(target_middle) or isinstance(target_middle, float) or pd.isna(middle_name) or isinstance(middle_name, float)) or
            (target_middle == "nan" or middle_name == "nan") or
            (target_middle == "NaN" or middle_name == "NaN")) or \
            (target_middle == "" or middle_name == ""):
            middle_lev = 0
            middle_jac = 0
            middle_init = 0
        else:
            middle_lev = Levenshtein.distance(middle_name, target_middle)
            middle_lev = 1 - (middle_lev / max(len(middle_name), len(target_middle))) if max(len(middle_name), len(target_middle)) > 0 else 0
            middle_jac = jaccard_similarity(middle_name, target_middle)
            middle_init = 1 if middle_name[0] == target_middle[0] else 0
        results["middle_lev"] = middle_lev
        results["middle_jac"] = middle_jac
        results["middle_init"] = middle_init

        # check if publication year is in lifespan
        lifespan_checks = []
        if ((pd.isna(target_birth_year) or pd.isna(target_death_year)) or
            (isinstance(target_birth_year, float) or isinstance(target_death_year, float)) or
            (target_birth_year == "nan" or target_death_year == "nan") or
            (target_birth_year == "NaN" or target_death_year == "NaN")) or \
            (target_birth_year == "" or target_death_year == ""):
            lifespan_checks.append(False)
        else:
            if source_year >= target_birth_year:
                lifespan_checks.append(True)
            else:
                lifespan_checks.append(False)
            if source_year <= target_death_year:
                lifespan_checks.append(True)
        if ((pd.isna(target_floruit_years[0]) or pd.isna(target_floruit_years[1])) or
            (isinstance(target_floruit_years[0], float) or isinstance(target_floruit_years[1], float)) or
            (target_floruit_years[0] == "nan" or target_floruit_years[1] == "nan") or
            (target_floruit_years[0] == "NaN" or target_floruit_years[1] == "NaN")) or \
            (target_floruit_years[0] == "" or target_floruit_years[1] == ""):
            lifespan_checks.append(False)
        else:
            if source_year >= target_floruit_years[0]:
                lifespan_checks.append(True)
            else:
                lifespan_checks.append(False)
            if source_year <= target_floruit_years[1]:
                lifespan_checks.append(True)

        # create a metric from the lifespan checks
        lifespan_metric = sum([1 for x in lifespan_checks if x]) / len(lifespan_checks) if len(lifespan_checks) > 0 else 0
        results["year"] = lifespan_metric

        # create a composite score using the weight arguments
        score = (weight["first_lev"] * first_lev +
                 weight["first_jac"] * first_jac +
                 weight["first_init"] * first_init +
                 weight["middle_lev"] * middle_lev +
                 weight["middle_jac"] * middle_jac +
                 weight["middle_init"] * middle_init +
                 weight["last_lev"] * last_lev +
                 weight["last_jac"] * last_jac +
                 weight["last_init"] * last_init +
                 weight["year"] * lifespan_metric)
        
        # normalize
        score = score / sum(weight.values())

        # add the score to the results
        results["score"] = score

        # return a tuple of the target name and the results
        return (df_row["name"], results["score"])
    
    # given a row, get names and scores from all rows in the dataframe
    # add them to a "prospective_names" list of tuples
    def _get_prospective_names(row, df, weights, n=10):
        prospective_names = []
        for index, df_row in df.iterrows():
            if index != row.name:
                prospective_names.append(_match_name(first_name, middle_name, last_name, year, df_row, weights))
        df = pd.DataFrame(prospective_names, columns=["name", "score"])
        df = df.sort_values(by="score", ascending=False)
        df = df.head(n)
        prospective_names = list(zip(df["name"], df["score"]))
        return prospective_names
    
    # get the prospective names
    prospective_names = _get_prospective_names(row, df, weights)

    # get the full author name
    author = row["author"]
    if not pd.isna(author) and not isinstance(author, float) and not author == "" and not author == "nan" and not author == "NaN":
        author = str(author)

    # return the prospective names
    return prospective_names
