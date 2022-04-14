from traceback import print_tb
from unittest import result
import pandas as pd
import numpy as np
import math

# client_id,homebanking_active,has_homebanking,has_insurance_21,has_insurance_23,has_life_insurance_fixed_cap
# has_life_insurance_decreasing_cap,has_fire_car_other_insurance,has_personal_loan,has_mortgage_loan,has_current_account
# has_pension_saving,has_savings_account,has_savings_account_starter,has_current_account_starter,bal_insurance_21
# bal_insurance_23,cap_life_insurance_fixed_cap,cap_life_insurance_decreasing_cap,prem_fire_car_other_insurance,bal_personal_loan
# bal_mortgage_loan,bal_current_account,bal_pension_saving,bal_savings_account,bal_savings_account_starter,bal_current_account_starter
# visits_distinct_so,visits_distinct_so_areas,customer_since_all,customer_since_bank,customer_gender,customer_birth_date,customer_postal_code
# customer_occupation_code,customer_self_employed,customer_education,customer_children,customer_relationship,target

# Get the maximum date in column


def getMaxDate(dates):
    maxYear = int(dates[0].split('-')[0])
    maxMonth = int(dates[0].split('-')[1])
    for date in dates:
        if isinstance(date, str):
            if int(date.split('-')[0]) > maxYear:
                maxYear = int(date.split('-')[0])
                maxMonth = int(date.split('-')[1])
            if int(date.split('-')[0]) == maxYear:
                if int(date.split('-')[1]) > maxMonth:
                    maxYear = int(date.split('-')[0])
                    maxMonth = int(date.split('-')[1])
    return maxYear, maxMonth

# Get the minimum date in column


def getMinDate(dates):
    maxYear = int(dates[0].split('-')[0])
    maxMonth = int(dates[0].split('-')[1])
    for date in dates:
        if isinstance(date, str):
            if int(date.split('-')[0]) < maxYear:
                maxYear = int(date.split('-')[0])
                maxMonth = int(date.split('-')[1])
            if int(date.split('-')[0]) == maxYear:
                if int(date.split('-')[1]) < maxMonth:
                    maxYear = int(date.split('-')[0])
                    maxMonth = int(date.split('-')[1])
    return maxYear, maxMonth

# Returns value between 0 an 1 of data


def normalizeDate(date, max, min):
    maxMonths = max[0]*12+max[1]-1
    minMonths = min[0]*12+min[1]-1
    dateMonths = date[0]*12+date[1]-1
    return (dateMonths-minMonths)/(maxMonths-minMonths)

# Returns the normalized average of dates


def normalizedAverageDate(dates):
    min = getMinDate(dates)
    max = getMaxDate(dates)
    nonNanValues = len(dates) - (dates.isnull()).sum()
    sum = 0
    for date in dates:
        if isinstance(date, str):
            sum += normalizeDate((int(date.split('-')
                                 [0]), int(date.split('-')[1])), min, max)
    return float(sum)/float(nonNanValues)

# Returns 3 lists for the classes of the zip codes for Brussels, Flanders, Wallonia and other


def getRegionClass(postalCodes):
    Brussels = []
    Flanders = []
    Wallonia = []
    Other = []
    for code in postalCodes:
        # Brussels
        if ((code >= 1000) and (code <= 1212)) or ((code >= 1931) and (code <= 1950)):
            Brussels.append(1)
            Flanders.append(0)
            Wallonia.append(0)
            Other.append(0)
        # Flanders
        elif ((code >= 1500) and (code <= 4690)) or ((code >= 8000) and (code <= 9999)):
            Brussels.append(0)
            Flanders.append(1)
            Wallonia.append(0)
            Other.append(0)
        # Wallonia
        elif (code >= 4000) and (code <= 7970):
            Brussels.append(0)
            Flanders.append(0)
            Wallonia.append(1)
            Other.append(0)
        # Other
        else:
            Brussels.append(0)
            Flanders.append(0)
            Wallonia.append(0)
            Other.append(1)
    return Brussels, Flanders, Wallonia, Other

# Returns Normalized column


def normalizeColumn(column):
    a, b = 0, 1
    min, max = column.min(), column.max()
    return (column-min) / (max - min) * (b - a) + a

# Returns Classes for occupations


def getOccupationClass(occupations):
    mode = occupations.mode()
    result = [[], [], [], [], [], [], [], [], [], []]
    for occupation in occupations:
        if math.isnan(occupation):
            for i in range(10):
                if i == int(mode):
                    result[i].append(1)
                else:
                    result[i].append(0)
        else:
            for i in range(10):
                if i == int(occupation):
                    result[i].append(1)
                else:
                    result[i].append(0)
    return result

# Returns column with normalized and missing values


def getNormalizedOccupations(occupations):
    result = []
    min, max = occupations.min(), occupations.max()
    mode = occupations.mode()
    for occupation in occupations:
        if math.isnan(occupation):
            result.append(float((mode-min) / (max - min)))
        else:
            result.append((occupation-min) / (max - min))
    return result

# Prints the percentages of missing data


def printMissingData(df):
    features = df.columns.values.tolist()

    print("Number of rows:", len(df))
    print("Missing data")
    print("------------")
    for feature in features:
        missing = (df[feature].isnull()).sum()
        if missing != 0:
            print(feature, ':', missing, '->', (missing/len(df))*100, '%')

# Prints the max and min of all data


def printMaxMinData(df):
    features = df.columns.values.tolist()

    print("Range of data")
    print("-------------")
    for feature in features:
        if not isinstance(df[feature][0], str) and feature != 'customer_children' and feature != 'customer_relationship':
            print(feature, ': min->',
                  df[feature].min(), 'max->', df[feature].max())
    print()

# Returns list with all normalized dates, changing the NaN values for the avg


def getNormalizedDates(dates):
    min = getMinDate(dates)
    max = getMaxDate(dates)
    average = normalizedAverageDate(dates)

    result = []
    for date in dates:
        if isinstance(date, str):
            normalDate = normalizeDate(
                (int(date.split('-')[0]), int(date.split('-')[1])), min, max)
            if normalDate < 0 or normalDate > 1:
                print(date)
            result.append(normalDate)
        else:
            result.append(average)
    return result


if __name__ == '__main__':
    df = pd.read_csv('data/train_month_3_with_target.csv')

    # USE BELOW LINE TO PROCESS THE TEST DATA, DONT FORGET TO CHANGE THE OUTPUT FILE BELOW
    # df = pd.read_csv('data/test_month_3.csv')

    Brussels, Flanders, Wallonia, Other = getRegionClass(
        df['customer_postal_code'])

    occupations = getOccupationClass(df['customer_occupation_code'])

    printMissingData(df)

    # newDf = pd.DataFrame({
    #     'client_id':df['client_id'],
    #     'homebanking_active':df['homebanking_active'],
    #     'has_homebanking':df['has_homebanking'],
    #     'has_insurance_21':df['has_insurance_21'],
    #     'has_insurance_23':df['has_insurance_23'],
    #     'has_life_insurance_fixed_cap':df['has_life_insurance_fixed_cap'],
    #     'has_life_insurance_decreasing_cap':df['has_life_insurance_decreasing_cap'],
    #     'has_fire_car_other_insurance':df['has_fire_car_other_insurance'],
    #     'has_personal_loan':df['has_personal_loan'],
    #     'has_mortgage_loan':df['has_current_account'],
    #     'has_pension_saving':df['has_pension_saving'],
    #     'has_savings_account':df['has_savings_account'],
    #     'has_savings_account_starter':df['has_savings_account_starter'],
    #     'has_current_account_starter':df['has_current_account_starter'],
    #     'bal_insurance_21':normalizeColumn(df['bal_insurance_21']),
    #     'bal_insurance_23':normalizeColumn(df['bal_insurance_23']),
    #     'cap_life_insurance_fixed_cap':normalizeColumn(df['cap_life_insurance_fixed_cap']),
    #     'cap_life_insurance_decreasing_cap':normalizeColumn(df['cap_life_insurance_decreasing_cap']),
    #     'prem_fire_car_other_insurance':normalizeColumn(df['prem_fire_car_other_insurance']),
    #     'bal_personal_loan':normalizeColumn(df['bal_personal_loan']),
    #     'bal_mortgage_loan':normalizeColumn(df['bal_mortgage_loan']),
    #     'bal_current_account':normalizeColumn(df['bal_current_account']),
    #     'bal_pension_saving':normalizeColumn(df['bal_pension_saving']),
    #     'bal_savings_account':normalizeColumn(df['bal_savings_account']),
    #     'bal_savings_account_starter':normalizeColumn(df['bal_savings_account_starter']),
    #     'bal_current_account_starter':normalizeColumn(df['bal_current_account_starter']),
    #     'visits_distinct_so':normalizeColumn(df['visits_distinct_so']),
    #     'visits_distinct_so_areas':normalizeColumn(df['visits_distinct_so_areas']),
    #     'customer_since_all':getNormalizedDates(df['customer_since_all']),
    #     'customer_since_bank':getNormalizedDates(df['customer_since_bank']),
    #     'customer_gender':normalizeColumn(df['customer_gender']),
    #     'customer_birth_date':getNormalizedDates(df['customer_birth_date']),
    #     'brussels_postal_code':Brussels,
    #     'flanders_postal_code':Flanders,
    #     'wallonia_postal_code':Wallonia,
    #     'other_postal_code':Other,
    #     'customer_occupation_code_0':occupations[0],
    #     'customer_occupation_code_1':occupations[1],
    #     'customer_occupation_code_2':occupations[2],
    #     'customer_occupation_code_3':occupations[3],
    #     'customer_occupation_code_4':occupations[4],
    #     'customer_occupation_code_5':occupations[5],
    #     'customer_occupation_code_6':occupations[6],
    #     'customer_occupation_code_7':occupations[7],
    #     'customer_occupation_code_8':occupations[8],
    #     'customer_occupation_code_9':occupations[9],
    #     'customer_self_employed':df['customer_self_employed'],
    #     'target':df['target'],
    # })

    newDf = pd.DataFrame({
        'client_id': df['client_id'],
        'homebanking_active': df['homebanking_active'],
        'has_homebanking': df['has_homebanking'],
        'has_insurance_21': df['has_insurance_21'],
        'has_insurance_23': df['has_insurance_23'],
        'has_life_insurance_fixed_cap': df['has_life_insurance_fixed_cap'],
        'has_life_insurance_decreasing_cap': df['has_life_insurance_decreasing_cap'],
        'has_fire_car_other_insurance': df['has_fire_car_other_insurance'],
        'has_personal_loan': df['has_personal_loan'],
        'has_mortgage_loan': df['has_current_account'],
        'has_pension_saving': df['has_pension_saving'],
        'has_savings_account': df['has_savings_account'],
        'has_savings_account_starter': df['has_savings_account_starter'],
        'has_current_account_starter': df['has_current_account_starter'],
        'bal_insurance_21': normalizeColumn(df['bal_insurance_21']),
        'bal_insurance_23': normalizeColumn(df['bal_insurance_23']),
        'cap_life_insurance_fixed_cap': normalizeColumn(df['cap_life_insurance_fixed_cap']),
        'cap_life_insurance_decreasing_cap': normalizeColumn(df['cap_life_insurance_decreasing_cap']),
        'prem_fire_car_other_insurance': normalizeColumn(df['prem_fire_car_other_insurance']),
        'bal_personal_loan': normalizeColumn(df['bal_personal_loan']),
        'bal_mortgage_loan': normalizeColumn(df['bal_mortgage_loan']),
        'bal_current_account': normalizeColumn(df['bal_current_account']),
        'bal_pension_saving': normalizeColumn(df['bal_pension_saving']),
        'bal_savings_account': normalizeColumn(df['bal_savings_account']),
        'bal_savings_account_starter': normalizeColumn(df['bal_savings_account_starter']),
        'bal_current_account_starter': normalizeColumn(df['bal_current_account_starter']),
        'visits_distinct_so': normalizeColumn(df['visits_distinct_so']),
        'visits_distinct_so_areas': normalizeColumn(df['visits_distinct_so_areas']),
        'customer_since_all': getNormalizedDates(df['customer_since_all']),
        'customer_since_bank': getNormalizedDates(df['customer_since_bank']),
        'customer_gender': normalizeColumn(df['customer_gender']),
        'customer_birth_date': getNormalizedDates(df['customer_birth_date']),
        'brussels_postal_code': Brussels,
        'flanders_postal_code': Flanders,
        'wallonia_postal_code': Wallonia,
        'other_postal_code': Other,
        'customer_occupation_code_0': getNormalizedOccupations(df['customer_occupation_code']),
        'customer_self_employed': df['customer_self_employed'],
        # COMMENT OUT THE BELOW LINE TO PROCESS THE TEST DATA
        'target': df['target'],
    })

    newDf.to_csv('out/cleanedDataClassOccupation.csv',
                 sep=',', encoding='utf-8', index=False)
    # USE BELOW TO PROCESS THE TEST DATA
    # newDf.to_csv('out/cleanedDataNoClassOccupation.csv',
    #              sep=',', encoding='utf-8', index=False)

    printMaxMinData(newDf)
