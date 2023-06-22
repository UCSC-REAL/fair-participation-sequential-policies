import numpy as np

# https://github.com/socialfoundations/folktables
from folktables import (
    BasicProblem,
    public_coverage_filter,
    adult_filter,
    travel_time_filter,
    ACSEmployment,
)


"""
We explicitly define the modified ACS problems here and note group changes from the original problems.

All class labels are 0/1 or False/True 
"""

# group: RAC1P -> PINCP (0: <=50k, 1: >50k)
Income = BasicProblem(
    features=[
        "AGEP",
        "COW",
        "SCHL",
        "MAR",
        "OCCP",
        "POBP",
        "RELP",
        "WKHP",
        "SEX",
        "RAC1P",
    ],
    target="PINCP",
    target_transform=lambda x: x > 50000,
    group="PINCP",
    group_transform=lambda x: x > 50000,
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

# filter to race classes 1/2 {0: 1(white), 1: 2(black)}
Employment = BasicProblem(
    features=[
        "AGEP",
        "SCHL",
        "MAR",
        "RELP",
        "DIS",
        "ESP",
        "CIT",
        "MIG",
        "MIL",
        "ANC",
        "NATIVITY",
        "DEAR",
        "DEYE",
        "DREM",
        "SEX",
        "RAC1P",
    ],
    target="ESR",
    target_transform=lambda x: x == 1,
    group="RAC1P",
    group_transform=lambda x: x == 2,
    preprocess=lambda x: x[x["RAC1P"].isin([1, 2])],
    postprocess=lambda x: np.nan_to_num(x, -1),
)

# group: RAC1P -> SEX {0: 1(Male), 1: 2(Female)}
PublicCoverage = BasicProblem(
    features=[
        "AGEP",
        "SCHL",
        "MAR",
        "SEX",
        "DIS",
        "ESP",
        "CIT",
        "MIG",
        "MIL",
        "ANC",
        "NATIVITY",
        "DEAR",
        "DEYE",
        "DREM",
        "PINCP",
        "ESR",
        "ST",
        "FER",
        "RAC1P",
    ],
    target="PUBCOV",
    target_transform=lambda x: x == 1,
    group="SEX",
    group_transform=lambda x: x == 2,
    preprocess=public_coverage_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)


# group: RAC1P -> AGEP {0: <=35, 1: >35}
TravelTime = BasicProblem(
    features=[
        "AGEP",
        "SCHL",
        "MAR",
        "SEX",
        "DIS",
        "ESP",
        "MIG",
        "RELP",
        "RAC1P",
        "PUMA",
        "ST",
        "CIT",
        "OCCP",
        "JWTR",
        "POWPUMA",
        "POVPIP",
    ],
    target="JWMNP",
    target_transform=lambda x: x > 20,
    group="AGEP",
    group_transform=lambda x: x > 35,
    preprocess=travel_time_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

# group: RAC1P -> MIG {0: >1, 1: 1}
Mobility = BasicProblem(
    features=[
        "AGEP",
        "SCHL",
        "MAR",
        "SEX",
        "DIS",
        "ESP",
        "CIT",
        "MIL",
        "ANC",
        "NATIVITY",
        "RELP",
        "DEAR",
        "DEYE",
        "DREM",
        "RAC1P",
        "GCL",
        "COW",
        "ESR",
        "WKHP",
        "JWMNP",
        "PINCP",
    ],
    target="MIG",
    target_transform=lambda x: x == 1,
    group="MIG",
    group_transform=lambda x: x == 1,
    preprocess=lambda x: x.drop(x.loc[(x["AGEP"] <= 18) | (x["AGEP"] >= 35)].index),
    postprocess=lambda x: np.nan_to_num(x, -1),
)

_names = ("Income", "Employment", "PublicCoverage", "TravelTime", "Mobility")
_states = ("AL", "TX", "AK", "CA", "FL")
_basic_problems = (Income, Employment, PublicCoverage, TravelTime, Mobility)
problems = dict()
for name, state, basic_problem in zip(_names, _states, _basic_problems):
    problems[name] = (basic_problem, [state])
