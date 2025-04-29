# Create a simple Python script for data analysis - Code

Created: 2025-03-11T22:36:02.899375

Here is a complete Python script for analyzing CSV files and producing statistical summaries. This script uses the `pandas`, `numpy`, and `matplotlib` libraries to read CSV files, perform data analysis, and visualize the results.

### Python Script: `data_analysis.py`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_csv(file_path):
    """
    Reads a CSV file and returns a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    DataFrame: A pandas DataFrame containing the CSV data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully read {file_path}")
        return df
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None

def analyze_data(df):
    """
    Analyzes the DataFrame and produces statistical summaries.
    
    Parameters:
    df (DataFrame): The DataFrame to analyze.
    
    Returns:
    dict: A dictionary containing statistical summaries.
    """
    summary = {}
    
    # Calculate mean, median, mode, and standard deviation for numerical columns
    for column in df.select_dtypes(include=[np.number]).columns:
        summary[column] = {
            'mean': df[column].mean(),
            'median': df[column].median(),
            'mode': df[column].mode()[0],  # mode can return multiple values, take the first
            'std_dev': df[column].std(),
            'count': df[column].count(),
            'min': df[column].min(),
            'max': df[column].max()
        }
    
    return summary

def visualize_data(df):
    """
    Visualizes the data using histograms and box plots for numerical columns.
    
    Parameters:
    df (DataFrame): The DataFrame to visualize.
    """
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    
    for column in numerical_columns:
        plt.figure(figsize=(12, 5))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(df[column], bins=30, color='blue', alpha=0.7)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        
        # Box Plot
        plt.subplot(1, 2, 2)
        plt.boxplot(df[column])
        plt.title(f'Box Plot of {column}')
        plt.ylabel(column)
        
        plt.tight_layout()
        plt.show()

def main(file_path):
    """
    Main function to execute the data analysis.
    
    Parameters:
    file_path (str): The path to the CSV file.
    """
    df = read_csv(file_path)
    if df is not None:
        summary = analyze_data(df)
        print("Statistical Summary:")
        for column, stats in summary.items():
            print(f"\nColumn: {column}")
            for stat_name, value in stats.items():
                print(f"{stat_name.capitalize()}: {value}")
        
        visualize_data(df)

if __name__ == "__main__":
    # Example usage: replace 'your_file.csv' with the path to your CSV file
    main('your_file.csv')
```

### Instructions for Use

1. **Install Required Libraries:**
   Make sure you have Python installed on your machine. You can install the required libraries using pip:

   ```bash
   pip install pandas numpy matplotlib
   ```

2. **Save the Script:**
   Save the above script as `data_analysis.py`.

3. **Prepare Your CSV File:**
   Ensure you have a CSV file ready for analysis. Replace `'your_file.csv'` in the `main` function call with the path to your CSV file.

4. **Run the Script:**
   Execute the script from the command line:

   ```bash
   python data_analysis.py
   ```

5. **View Results:**
   The script will print statistical summaries for each numerical column in the CSV file and display histograms and box plots for visual analysis.

### Features of the Script

- **CSV Reading:** The script reads a CSV file and handles errors if the file cannot be read.
- **Statistical Analysis:** It calculates mean, median, mode, standard deviation, count, minimum, and maximum for each numerical column.
- **Data Visualization:** It generates histograms and box plots for each numerical column to visualize the distribution and identify outliers.
- **Modular Design:** The script is organized into functions for better readability and maintainability.

### Testing the Script

To ensure the script works correctly, you can create a sample CSV file with numerical data. Here’s an example of how the CSV file might look:

```csv
Name,Age,Salary
Alice,30,70000
Bob,25,50000
Charlie,35,80000
David,40,120000
Eve,29,60000
```

Save this as `sample_data.csv` and replace `'your_file.csv'` in the script with `'sample_data.csv'` to test the functionality.

### Conclusion

This script provides a comprehensive solution for analyzing CSV files and generating statistical summaries. You can further enhance it by adding features such as handling missing values, analyzing categorical data, or exporting results to a new CSV file.