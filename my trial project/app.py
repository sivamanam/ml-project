import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Step 1: Loading the dataset from the CSV file
df = pd.read_csv('students_data.csv')  # Ensure your CSV file path is correct

# this is for column names checking having correctly or not
if 'skills' not in df.columns or 'lacking skills' not in df.columns or 'recommended jobs' not in df.columns:
    raise ValueError("CSV file must contain 'skills', 'lacking skills', and 'recommended jobs' columns.")

# Combine skills and lacking skills into a single string per job for vectorization
df['combined_skills'] = df['skills'].astype(str) + ',' + df['lacking skills'].astype(str)

# Step 2: Text vectorization using TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['combined_skills'])

# Step 3: my algorithm NearestNeighbors for recommendations
model = NearestNeighbors(n_neighbors=3, metric='cosine')
model.fit(X)

# Step 4: Function to recommend jobs based on user input skills
def recommend_job(user_skills):
    # Vectorize user input
    user_skills_vector = vectorizer.transform([user_skills])
    
    # Finding the nearest jobs
    distances, indices = model.kneighbors(user_skills_vector)

    # Printing recommended job (closest match)
    recommended_job = df.iloc[indices[0][0]]['recommended jobs']
    print(f"\nRecommended Job: {recommended_job}")

    # Print related job recommendations (other close matches)
    print("\nRelated Jobs:")
    for idx in indices[0]:
        job = df.iloc[idx]
        print(f"- {job['recommended jobs']}")
        
        # Extracting and printing the missing skills for the related jobs
        missing_skills = job['lacking skills']
        print(f"  Missing Skills: {missing_skills}")

# Main function to taking user input and making recommendations
def main():
    # Asking the user to input their skills
    user_input_skills = input("Enter your skills separated by commas: ")
    recommend_job(user_input_skills)

if __name__ == "__main__":
    main()
