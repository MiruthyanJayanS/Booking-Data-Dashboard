import pandas as pd

def load_data(file_path, sheet_name):
    """Load the Excel file and return the specified sheet as a DataFrame."""
    xls = pd.ExcelFile(file_path)
    return pd.read_excel(xls, sheet_name=sheet_name)

def analyze_missing_values(df):
    """Analyze missing values in the DataFrame."""
    missing_values = df.isnull().sum()
    missing_values_percentage = (missing_values / len(df)) * 100
    missing_summary = pd.DataFrame({
        'Missing Values': missing_values, 
        'Percentage': missing_values_percentage.round(2)
    }).sort_values(by='Missing Values', ascending=False)
    return missing_summary

def clean_data(df):
    """Perform data cleaning on the given DataFrame."""
    df_clean = df.drop(columns=["Subscription Type"], errors='ignore')
    
    # Step 1: Fill missing 'Instructor' and 'Time Slot' based on 'Class Type'
    df_clean['Instructor'] = df_clean.groupby('Class Type')['Instructor'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown')
    )
    df_clean['Time Slot'] = df_clean.groupby('Class Type')['Time Slot'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown')
    )
    
    # Step 2: Fill missing 'Duration (mins)' based on median of 'Class Type'
    df_clean['Duration (mins)'] = df_clean.groupby('Class Type')['Duration (mins)'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Step 3: Fill missing 'Facility' based on most common facility per 'Service Name'
    df_clean['Facility'] = df_clean.groupby('Service Name')['Facility'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Not Assigned')
    )
    
    # Step 4: Fill missing 'Theme' for 'Birthday Party' bookings
    df_clean.loc[df_clean['Booking Type'] == 'Birthday Party', 'Theme'] = df_clean['Theme'].fillna('No Theme Specified')

    # Step 5: Extract year-month for trend analysis
    df_clean['Year-Month'] = df_clean['Booking Date'].dt.to_period('M')
    
    return df_clean

def save_data(df, output_path):
    """Save cleaned DataFrame to a CSV file."""
    df.to_csv(output_path, index=False)

def main():
    file_path = r"E:\Project\Omnify\DataAnalyst_Assesment_Dataset.xlsx"
    sheet_name = 'Large_Fake_Bookings_With_Discre'
    output_path = "cleaned_file.csv"
    
    print("Loading data...")
    df = load_data(file_path, sheet_name)
    
    print("Analyzing missing values...")
    missing_summary = analyze_missing_values(df)
    print(missing_summary)
    
    print("Cleaning data...")
    df_clean = clean_data(df)
    
    print("Saving cleaned data to file...")
    save_data(df_clean, output_path)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    main()
