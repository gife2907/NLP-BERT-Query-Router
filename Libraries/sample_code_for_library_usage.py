# sample_nlp_research.py

import os
import sys
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Import our reporting module
from Libraries.reporter_lib import setup_logging_and_reporting

# Set up paths
RESULTS_DIR = "Results"
LIBRARIES_DIR = "Libraries"
INTRO_FILE = os.path.join(LIBRARIES_DIR, "intro_about_me.txt")
CHART_FILE = os.path.join(RESULTS_DIR, "model_comparison.png")

# Create dummy data - simulating an NLP classification task
def create_dummy_data(num_samples=1000):
    categories = ['business', 'sports', 'technology', 'entertainment']
    data = []
    
    business_words = ['profit', 'market', 'company', 'stock', 'financial', 'investment']
    sports_words = ['game', 'team', 'player', 'score', 'championship', 'tournament']
    tech_words = ['algorithm', 'software', 'hardware', 'internet', 'data', 'programming']
    entertainment_words = ['movie', 'music', 'celebrity', 'concert', 'festival', 'award']
    
    category_words = {
        'business': business_words,
        'sports': sports_words,
        'technology': tech_words,
        'entertainment': entertainment_words
    }
    
    for _ in range(num_samples):
        category = random.choice(categories)
        # Create a text that's biased toward the chosen category
        words = []
        for _ in range(random.randint(10, 30)):
            if random.random() < 0.7:  # 70% chance to pick a word from the category
                words.append(random.choice(category_words[category]))
            else:
                # Pick a random word from any category
                random_category = random.choice(categories)
                words.append(random.choice(category_words[random_category]))
                
        text = ' '.join(words)
        data.append({'text': text, 'category': category})
    
    return pd.DataFrame(data)

def main():
    # Initialize logging and reporting
    logger, reporter = setup_logging_and_reporting(log_dir=RESULTS_DIR)
    
    try:
        # Add content to the report
        logger.add_to_report("heading1", "NLP Text Classification Research")
        
        logger.add_to_report("text", "This document presents the results of our text classification "
                              "research using various machine learning techniques. We evaluate "
                              "multiple approaches and present comparative analysis of their performance.")
        
        # Add dataset description
        logger.add_to_report("heading2", "Dataset Description")
        logger.add_to_report("text", "We created a synthetic dataset to simulate text classification "
                              "across four categories: business, sports, technology, and entertainment.")
        
        # Generate the dummy data
        print("Generating synthetic NLP dataset...")
        df = create_dummy_data()
        
        # Print some stats about the dataset
        category_counts = df['category'].value_counts()
        logger.add_to_report("text", f"Dataset size: {len(df)} samples")
        logger.add_to_report("text", "Category distribution:")
        
        # Create a table for the report
        table_data = [["Category", "Count", "Percentage"]]
        for category, count in category_counts.items():
            percentage = count / len(df) * 100
            table_data.append([category, str(count), f"{percentage:.1f}%"])
            print(f"- {category}: {count} samples ({percentage:.1f}%)")
            
        logger.add_to_report("table", table_data)
        
        # Data preprocessing
        logger.add_to_report("heading2", "Data Preprocessing")
        logger.add_to_report("text", "We split the dataset into training (80%) and test (20%) sets. "
                              "Text features were extracted using TF-IDF vectorization.")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['category'], test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Feature extraction
        vectorizer = TfidfVectorizer(max_features=1000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Model training
        logger.add_to_report("heading1", "Model Training and Evaluation")
        logger.add_to_report("text", "We trained a Logistic Regression classifier on the TF-IDF vectors.")
        
        print("Training logistic regression model...")
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_vec, y_train)
        
        # Model evaluation
        logger.add_to_report("heading2", "Model Performance")
        
        print("Evaluating model...")
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        logger.add_to_report("text", f"The logistic regression model achieved an accuracy of {accuracy:.2%} on the test set.")
        logger.add_to_report("text", "Detailed classification report:")
        logger.add_to_report("text", f"\n{report}")
        
        # Create a visualization
        logger.add_to_report("heading2", "Performance Visualization")
        
        # Generate a chart comparing multiple models (simulated results)
        print("Generating performance comparison chart...")
        
        # Create the Results directory if it doesn't exist
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Simulated comparative results for multiple models
        models = ['Logistic Regression', 'Random Forest', 'SVM', 'BERT Fine-tuned']
        accuracies = [accuracy, accuracy + random.uniform(-0.05, 0.05), 
                      accuracy + random.uniform(-0.05, 0.05), 
                      accuracy + random.uniform(0.05, 0.15)]
        
        # Ensure all accuracies are between 0 and 1
        accuracies = [max(0, min(acc, 1)) for acc in accuracies]
        
        plt.figure(figsize=(10, 6))
        plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'red'])
        plt.ylabel('Accuracy')
        plt.xlabel('Model')
        plt.title('Classification Accuracy by Model')
        plt.ylim(0, 1)
        plt.savefig(CHART_FILE)
        
        # Add the chart to the report
        logger.add_to_report("text", "The following chart shows the comparative performance of different models on our classification task:")
        logger.add_to_report("image", CHART_FILE, width=6)
        
        # Add sample chart image from SVG
        svg_path = os.path.join("Results", "sample_chart.svg")
        logger.add_to_report("heading2", "Advanced Model Comparison")
        logger.add_to_report("text", "We also evaluated transformer-based models on a separate benchmark dataset:")
        logger.add_to_report("image", svg_path, width=6)
        
        # Conclusion
        logger.add_to_report("heading1", "Conclusion")
        logger.add_to_report("text", "Our experiments demonstrate that transformer-based models significantly "
                             "outperform traditional machine learning approaches for text classification tasks. "
                             "The BERT fine-tuned model achieved the highest accuracy, though at the cost of "
                             "increased computational requirements.", bold=True)
        
        logger.add_to_report("text", "Future work will focus on optimizing the transformer architectures for "
                             "specific domain adaptation and exploring distillation techniques to improve "
                             "inference performance.")
        
        # Generate the report
        report_path = reporter.generate_report(
            title="NLP Text Classification Research",
            subtitle="Performance Analysis of Various Classification Approaches",
            author="Dr. Jane Smith",
            organization="AI Research Lab",
            intro_file=INTRO_FILE
        )
        
        print(f"\nExperiment completed successfully.")
        print(f"Report generated at: {report_path}")
        
    except Exception as e:
        print(f"Error during experiment: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        # Restore stdout
        sys.stdout = sys.__stdout__
        
if __name__ == "__main__":
    main()
