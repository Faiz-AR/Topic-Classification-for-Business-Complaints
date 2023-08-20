import pandas as pd

def clean_data(filepath: str) -> pd.DataFrame:
    """
    Parameters
    ----------
    filepath : str
        Takes in a filepath to the dataset.

    Returns
    -------
    df : pd.DataFrame
        Returns a pandas dataframe which has been cleaned.

    """
    # Load the CFPB Consumer Complain dataset
    df = pd.read_csv(filepath)

    # Extract only the Product and Consumer complain narative column and filter
    # out the rest of the columns in the dataframe
    df = df[['Product', 'Consumer complaint narrative']]

    # Rename the columns into category and complaint
    df = df.rename(columns={
        'Product': 'complaint_category',
        'Consumer complaint narrative': 'complaint_narrative'
    })

    # Dictionary to minimize and shorten the complaints categories into 5
    category_dict = {
        'Credit reporting, credit repair services, or other personal consumer reports': 'credit_reporting',
        'Debt collection': 'debt_collection',
        'Credit card or prepaid card': 'credit_card',
        'Checking or savings account': 'retail_banking',
        'Money transfer, virtual currency, or money service': 'retail_banking',
        'Vehicle loan or lease': 'mortgages_and_loans',
        'Mortgage': 'mortgages_and_loans',
        'Student loan': 'mortgages_and_loans',
        'Payday loan, title loan, or personal loan': 'mortgages_and_loans'
    }

    # Replace the category into 5: credit_reporting, debt_collection,
    # credit_card, retail_banking and mortgages_and_loans
    df['complaint_category'].replace(category_dict, inplace=True)

    # Remove complaints with NaN values
    df.dropna(inplace=True)

    # Remove blank complaints
    blanks = []
    for index, category, complaint in df.itertuples():
        if type(complaint) == str:
            if complaint.isspace():
                blanks.append(index)

    df.drop(blanks, inplace=True)

    return df