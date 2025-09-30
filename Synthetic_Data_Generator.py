import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import string

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
n_rows = 15000
n_users = 150
start_date = datetime(2021, 1, 1)
end_date = datetime(2024, 12, 31)

# User IDs
user_ids = [f"U{str(i).zfill(3)}" for i in range(1, n_users + 1)]

# Categories with typos and variations
categories_clean = ['Food', 'Rent', 'Travel', 'Utilities', 'Health', 'Education', 'Entertainment', 'Savings', 'Others']
categories_with_typos = {
    'Food': ['Food', 'Foodd', 'food', 'FOOD', 'Fod', 'Foods'],
    'Rent': ['Rent', 'rent', 'RENT', 'Rnt', 'Rentt'],
    'Travel': ['Travel', 'Traval', 'travel', 'Travl', 'TRAVEL'],
    'Utilities': ['Utilities', 'Utilties', 'Utlities', 'utilities', 'Utility'],
    'Health': ['Health', 'health', 'Helth', 'HEALTH'],
    'Education': ['Education', 'education', 'Educaton', 'EDU'],
    'Entertainment': ['Entertainment', 'entertainment', 'Entrtnmnt', 'Entertain'],
    'Savings': ['Savings', 'savings', 'Saving', 'SAVINGS'],
    'Others': ['Others', 'others', 'Other', 'OTHERS', 'Misc']
}

# Payment modes with variations
payment_modes_clean = ['Cash', 'Card', 'UPI', 'Bank Transfer']
payment_modes_with_typos = {
    'Cash': ['Cash', 'cash', 'Csh', 'CASH', 'csh'],
    'Card': ['Card', 'card', 'CRD', 'CARD', 'Crd'],
    'UPI': ['UPI', 'upi', 'Upi', 'UPi'],
    'Bank Transfer': ['Bank Transfer', 'Bank Transfr', 'bank transfer', 'BankTransfer', 'Bank_Transfer']
}

# Locations with variations
locations = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']
locations_with_variations = []
for loc in locations:
    locations_with_variations.extend([loc, loc.lower(), loc.upper(), loc[:3].upper()])

# Sample notes
notes_samples = [
    'Grocery shopping', 'Monthly rent payment', 'Uber ride', 'Electricity bill', 'Doctor consultation',
    'Online course', 'Movie tickets', 'Restaurant dinner', 'Fixed deposit', 'Shopping',
    'Petrol', 'Train ticket', 'Medicine', 'Books', 'Gym membership', 'Internet bill',
    'Coffee', 'Lunch', 'ATM withdrawal', 'EMI payment', None, '', 'N/A', 'misc',
    'xyz123', 'test', '!!!', '...', 'asdfgh', None, None, None  # Gibberish and nulls
]

# Generate dataset
data = []
transaction_counter = 1

for _ in range(n_rows):
    # Transaction ID (with some duplicates)
    if random.random() < 0.07:  # 7% chance of duplicate
        if len(data) > 0:
            transaction_id = random.choice([d['transaction_id'] for d in data[-min(100, len(data)):]])
        else:
            transaction_id = f"T{str(transaction_counter).zfill(4)}"
    else:
        transaction_id = f"T{str(transaction_counter).zfill(4)}"
        transaction_counter += 1
    
    # User ID
    user_id = random.choice(user_ids)
    
    # Date (with various formats and missing values)
    if random.random() < 0.03:  # 3% missing dates
        date = None
    else:
        days_offset = random.randint(0, (end_date - start_date).days)
        date_obj = start_date + timedelta(days=days_offset)
        
        # Random date format
        format_choice = random.random()
        if format_choice < 0.4:
            date = date_obj.strftime('%Y-%m-%d')
        elif format_choice < 0.7:
            date = date_obj.strftime('%m/%d/%Y')
        elif format_choice < 0.9:
            date = date_obj.strftime('%d-%m-%y')
        else:
            date = date_obj.strftime('%d-%m-%Y')
    
    # Transaction type (85% Expense, 15% Income)
    if random.random() < 0.85:
        transaction_type = 'Expense'
    else:
        transaction_type = 'Income'
    
    # Category (with imbalance and typos)
    if random.random() < 0.02:  # 2% missing categories
        category = None
    else:
        if transaction_type == 'Income':
            category = random.choice(['Salary', 'Freelance', 'Investment', 'Others', 'Bonus'])
        else:
            # Weighted selection for expenses (more Food and Rent)
            weights = [0.25, 0.20, 0.15, 0.12, 0.05, 0.08, 0.10, 0.03, 0.02]
            category_base = np.random.choice(categories_clean, p=weights)
            category = random.choice(categories_with_typos[category_base])
    
    # Amount (with outliers, negatives, missing, and currency symbols)
    if random.random() < 0.02:  # 2% missing amounts
        amount = None
    else:
        if transaction_type == 'Income':
            # Income amounts (generally higher)
            if random.random() < 0.05:  # 5% outliers
                amount = random.randint(100000, 999999)
            else:
                amount = random.randint(10000, 80000)
        else:
            # Expense amounts
            if category and 'Rent' in category:
                amount = random.randint(5000, 50000)
            elif category and 'Food' in category:
                amount = random.randint(50, 5000)
            else:
                amount = random.randint(100, 10000)
            
            # Add outliers
            if random.random() < 0.03:
                amount = random.choice([999999, -500, -1000, 0, 999999999])
        
        # Add currency symbols randomly
        currency_choice = random.random()
        if currency_choice < 0.1:
            amount = f"₹{amount}"
        elif currency_choice < 0.15:
            amount = f"${amount}"
        elif currency_choice < 0.18:
            amount = f"Rs.{amount}"
        elif currency_choice < 0.2:
            amount = f"{amount} INR"
        # else keep as numeric
    
    # Payment mode
    if random.random() < 0.05:  # 5% missing payment modes
        payment_mode = None
    else:
        mode_base = random.choice(payment_modes_clean)
        payment_mode = random.choice(payment_modes_with_typos[mode_base])
    
    # Location
    if random.random() < 0.08:  # 8% missing locations
        location = random.choice([None, '', 'N/A', np.nan])
    else:
        location = random.choice(locations_with_variations)
    
    # Notes
    notes = random.choice(notes_samples)
    
    # Append row
    data.append({
        'transaction_id': transaction_id,
        'user_id': user_id,
        'date': date,
        'transaction_type': transaction_type,
        'category': category,
        'amount': amount,
        'payment_mode': payment_mode,
        'location': location,
        'notes': notes
    })

# Create DataFrame
df = pd.DataFrame(data)

# Add some complete duplicate rows (5-8%)
n_duplicates = int(len(df) * 0.06)
duplicate_indices = np.random.choice(df.index, n_duplicates, replace=True)
duplicate_rows = df.loc[duplicate_indices].copy()
df = pd.concat([df, duplicate_rows], ignore_index=True)

# Shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)

# Save to CSV
df.to_csv('budgetwise_finance_dataset.csv', index=False)

# Display first 20 rows and dataset info
print("BUDGETWISE PERSONAL FINANCE DATASET")
print("=" * 80)
print(f"\nDataset Shape: {df.shape}")
print(f"Total Rows: {len(df)}")
print(f"Unique Users: {df['user_id'].nunique()}")
print(f"Date Range: {df['date'].dropna().min()} to {df['date'].dropna().max()}")
print(f"\nTransaction Type Distribution:")
print(df['transaction_type'].value_counts())
print(f"\nCategory Distribution (top 10):")
print(df['category'].value_counts().head(10))
print(f"\nMissing Values:")
print(df.isnull().sum())
print(f"\nDuplicate Rows: {df.duplicated().sum()}")
print("\n" + "=" * 80)
print("FIRST 20 ROWS (showing data messiness):")
print("=" * 80)
print(df.head(20).to_string())

# Save sample for verification
print("\n" + "=" * 80)
print("CSV FORMAT (First 10 rows):")
print("=" * 80)
print(df.head(10).to_csv(index=False))