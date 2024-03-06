import os, re
import pandas as pd
books_path = "./data/MWTGBookletsExcel/"
booklets = os.listdir(books_path)


# Function to clean text
def clean_text(text):
    # Remove unicode characters
    cleaned_text = text.encode('ascii', 'ignore').decode()
    # Convert to lowercase
    #cleaned_text = cleaned_text.lower()
    # Remove special characters and punctuation
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_text)
    # Remove extra whitespaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    # Other cleaning steps if needed
    return cleaned_text


if __name__ == "__main__":
    for i in booklets:
        df = pd.read_excel(f"{books_path}/{i}", names=["paragraph", "text"])
        # Clean the text column
        df['text'] = df['text'].astype("str")
        df['text'] = df['text'].apply(clean_text)

        # Drop the paragraph column
        df.drop('paragraph', axis=1, inplace=True)
        df.drop_duplicates(keep='first')

        # Save cleaned dataframe to Excel
        output_file = f'./data/preprocessed/{i}'
        df.to_excel(output_file, index=False)
        
        
        print("Cleaned data saved to", output_file)