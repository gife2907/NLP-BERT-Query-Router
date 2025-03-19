import yaml
import pandas as pd
import numpy as np
import time
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

# Import the reporting library
from Libraries.reporter_lib import setup_logging_and_reporting
LIBRARIES_DIR = "Libraries"
INTRO_FILE = os.path.join(LIBRARIES_DIR, "intro_about_me.txt")
GLOSSARY_FILE = os.path.join(LIBRARIES_DIR, "classification_glossary.txt")

# Define models to test
models_to_test = [
    'bert-base-uncased',
    'prajjwal1/bert-mini',
    'all-MiniLM-L6-v2',
   # 'microsoft/deberta-v3-base',
   # 'sentence-transformers/all-mpnet-base-v2',
   # 'sentence-transformers/all-roberta-large-v1'
]

# Create results directory if it doesn't exist
RESULTS_DIR = "Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_csv_data(csv_file_path: str) -> pd.DataFrame:
    """
    Load test data from CSV file with structure: query,category,subcategory
    
    Args:
        csv_file_path: Path to the CSV file containing test data
        
    Returns:
        DataFrame with columns: query, category, subcategory
    """
    return pd.read_csv(csv_file_path, skiprows=1, names=['query', 'category', 'subcategory'])

def load_topics(topics_file_path: str) -> pd.DataFrame:
    """
    Load topics from the YAML file into a DataFrame structure.
    
    Args:
        topics_file_path: Path to the topics YAML file
        
    Returns:
        DataFrame with topic classification, title, and description
    """
    # Load the YAML file
    with open(topics_file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    # Create a list to store all topics
    all_topics = []
    
    # Parse the nested structure
    for category, topics in data['topic_classification'].items():
        for topic in topics:
            all_topics.append({
                'category': category,
                'title': topic['title'],
                'description': topic['description']
            })
    
    # Convert to DataFrame
    return pd.DataFrame(all_topics)

def create_topic_texts(topics_df: pd.DataFrame) -> List[str]:
    """
    Create full text descriptions for topics to generate embeddings.
    
    Args:
        topics_df: DataFrame with topic information
        
    Returns:
        List of combined topic texts
    """
    return [f"Category: {row['category']} - Title: {row['title']} - {row['description']}" 
            for _, row in topics_df.iterrows()]

def classify_queries_with_alternatives(queries: List[str], 
                                     topic_embeddings: np.ndarray, 
                                     topics_df: pd.DataFrame,
                                     model_name: str) -> Tuple[List[Dict], float, float]:
    """
    Classify each query to the most similar topic and also provide second best option.
    
    Args:
        queries: List of query strings
        topic_embeddings: Embeddings of the topic descriptions
        topics_df: DataFrame containing the topic information
        model_name: Name of the model to use
        
    Returns:
        Tuple of:
        - List of dictionaries with query, predicted category/subcategory, and alternatives
        - Time to classify first 10 entries
        - Time to classify last 10 entries
    """
    # Initialize model
    model = SentenceTransformer(model_name)
    
    results = []
    
    # Measure time for first 10 entries
    start_time_first = time.time()
    first_10_queries = queries[:10]
    first_10_embeddings = model.encode(first_10_queries)
    
    for i, query in enumerate(first_10_queries):
        # Calculate similarity with all topics
        similarities = cosine_similarity([first_10_embeddings[i]], topic_embeddings)[0]
        
        # Get indices of top 2 similar topics
        top_indices = np.argsort(similarities)[-2:][::-1]
        
        # Get the predicted (top) topic
        predicted_topic = topics_df.iloc[top_indices[0]]
        
        # Get the second best topic
        second_best_topic = topics_df.iloc[top_indices[1]]
        
        # Calculate distance between first and second best cosines
        cosine_distance = similarities[top_indices[0]] - similarities[top_indices[1]]
        
        results.append({
            'query': query,
            'predicted_category': predicted_topic['category'],
            'predicted_subcategory': predicted_topic['title'],
            'predicted_description': predicted_topic['description'],
            'similarity': similarities[top_indices[0]],
            'second_best_category': second_best_topic['category'],
            'second_best_subcategory': second_best_topic['title'],
            'second_best_description': second_best_topic['description'],
            'second_best_similarity': similarities[top_indices[1]],
            'cosine_distance': cosine_distance
        })
    
    time_for_first_10 = time.time() - start_time_first
    
    # Process the rest of the queries (excluding the first 10)
    middle_queries = queries[10:-10]
    if middle_queries:
        middle_embeddings = model.encode(middle_queries)
        
        for i, query in enumerate(middle_queries):
            similarities = cosine_similarity([middle_embeddings[i]], topic_embeddings)[0]
            top_indices = np.argsort(similarities)[-2:][::-1]
            
            predicted_topic = topics_df.iloc[top_indices[0]]
            second_best_topic = topics_df.iloc[top_indices[1]]
            cosine_distance = similarities[top_indices[0]] - similarities[top_indices[1]]
            
            results.append({
                'query': query,
                'predicted_category': predicted_topic['category'],
                'predicted_subcategory': predicted_topic['title'],
                'predicted_description': predicted_topic['description'],
                'similarity': similarities[top_indices[0]],
                'second_best_category': second_best_topic['category'],
                'second_best_subcategory': second_best_topic['title'],
                'second_best_description': second_best_topic['description'],
                'second_best_similarity': similarities[top_indices[1]],
                'cosine_distance': cosine_distance
            })
    
    # Measure time for last 10 entries
    start_time_last = time.time()
    last_10_queries = queries[-10:]
    last_10_embeddings = model.encode(last_10_queries)
    
    for i, query in enumerate(last_10_queries):
        similarities = cosine_similarity([last_10_embeddings[i]], topic_embeddings)[0]
        top_indices = np.argsort(similarities)[-2:][::-1]
        
        predicted_topic = topics_df.iloc[top_indices[0]]
        second_best_topic = topics_df.iloc[top_indices[1]]
        cosine_distance = similarities[top_indices[0]] - similarities[top_indices[1]]
        
        results.append({
            'query': query,
            'predicted_category': predicted_topic['category'],
            'predicted_subcategory': predicted_topic['title'],
            'predicted_description': predicted_topic['description'],
            'similarity': similarities[top_indices[0]],
            'second_best_category': second_best_topic['category'],
            'second_best_subcategory': second_best_topic['title'],
            'second_best_description': second_best_topic['description'],
            'second_best_similarity': similarities[top_indices[1]],
            'cosine_distance': cosine_distance
        })
    
    time_for_last_10 = time.time() - start_time_last
    
    return results, time_for_first_10, time_for_last_10

def create_confusion_matrix(y_true: List[str], y_pred: List[str], class_names: List[str], file_path: str) -> np.ndarray:
    """
    Create a confusion matrix and save it as an image.
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        class_names: List of class names
        file_path: Path to save the confusion matrix image
        
    Returns:
        Confusion matrix as numpy array
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()  # Close the figure to avoid display popup
    
    return cm

def calculate_metrics(y_true: List[str], y_pred: List[str], class_names: List[str]) -> Dict:
    """
    Calculate accuracy, precision, recall, and F1 score.
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        class_names: List of class names
        
    Returns:
        Dictionary with calculated metrics
    """
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate precision, recall, and F1 score (weighted average)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics

def truncate_text(text: str, max_length: int) -> str:
    """
    Truncate text to a specified maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of the truncated text
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    else:
        return text[:max_length-3] + "..."

def get_category_errors(results: List[Dict], true_categories: List[str], true_subcategories: List[str], 
                         topics_df: pd.DataFrame) -> List[Dict]:
    """
    Get entries where the predicted category is wrong.
    """
    category_errors = []
    
    for i, result in enumerate(results):
        if result['predicted_category'] != true_categories[i]:
            # Get the description of the true category/subcategory
            true_desc_row = topics_df[(topics_df['category'] == true_categories[i]) & 
                                     (topics_df['title'] == true_subcategories[i])]
            
            true_description = ""
            if not true_desc_row.empty:
                true_description = true_desc_row.iloc[0]['description']
            
            # Get the description of the second best category/subcategory
            second_best_desc_row = topics_df[(topics_df['category'] == result['second_best_category']) & 
                                           (topics_df['title'] == result['second_best_subcategory'])]
            
            second_best_description = ""
            if not second_best_desc_row.empty:
                second_best_description = second_best_desc_row.iloc[0]['description']
            
            category_errors.append({
                'query': result['query'],
                'true_category': true_categories[i],
                'true_subcategory': true_subcategories[i],
                'true_description': true_description,
                'predicted_category': result['predicted_category'],
                'predicted_subcategory': result['predicted_subcategory'],
                'predicted_description': result['predicted_description'],
                'similarity': result['similarity'],
                'second_best_category': result['second_best_category'],
                'second_best_subcategory': result['second_best_subcategory'],
                'second_best_description': second_best_description,  # Add this line
                'second_best_similarity': result['second_best_similarity'],
                'cosine_distance': result['cosine_distance']
            })
    
    return category_errors

def get_subcategory_errors(results: List[Dict], true_categories: List[str], true_subcategories: List[str], 
                           topics_df: pd.DataFrame) -> List[Dict]:
    """
    Get entries where the predicted category is correct but subcategory is wrong.
    """
    subcategory_errors = []
    
    for i, result in enumerate(results):
        if (result['predicted_category'] == true_categories[i] and 
            result['predicted_subcategory'] != true_subcategories[i]):
            
            # Get the description of the true category/subcategory
            true_desc_row = topics_df[(topics_df['category'] == true_categories[i]) & 
                                     (topics_df['title'] == true_subcategories[i])]
            
            true_description = ""
            if not true_desc_row.empty:
                true_description = true_desc_row.iloc[0]['description']
            
            # Get the description of the second best category/subcategory
            second_best_desc_row = topics_df[(topics_df['category'] == result['second_best_category']) & 
                                           (topics_df['title'] == result['second_best_subcategory'])]
            
            second_best_description = ""
            if not second_best_desc_row.empty:
                second_best_description = second_best_desc_row.iloc[0]['description']
            
            subcategory_errors.append({
                'query': result['query'],
                'true_category': true_categories[i],
                'true_subcategory': true_subcategories[i],
                'true_description': true_description,
                'predicted_category': result['predicted_category'],
                'predicted_subcategory': result['predicted_subcategory'],
                'predicted_description': result['predicted_description'],
                'similarity': result['similarity'],
                'second_best_category': result['second_best_category'],
                'second_best_subcategory': result['second_best_subcategory'],
                'second_best_description': second_best_description,  # Add this line
                'second_best_similarity': result['second_best_similarity'],
                'cosine_distance': result['cosine_distance']
            })
    
    return subcategory_errors

def create_spider_chart(model_metrics: Dict[str, Dict], file_path: str):
    """
    Create a spider chart comparing all models.
    
    Args:
        model_metrics: Dictionary with model names as keys and metrics as values
        file_path: Path to save the spider chart image
    """
    # Define the metrics to plot
    metrics = ['accuracy', 'precision', 'f1_score', 'queries_per_second']
    
    # Define labels for the metrics
    labels = ['Accuracy', 'Precision', 'F1 Score', 'Queries/Second']
    
    # Number of metrics
    num_metrics = len(metrics)
    
    # Number of models
    models = list(model_metrics.keys())
    num_models = len(models)
    
    # Create angles for each metric
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Define colors for each model
    colors = plt.cm.tab10(np.linspace(0, 1, num_models))
    
    # Plot each model
    for i, model in enumerate(models):
        values = [model_metrics[model][metric] for metric in metrics]
        values += values[:1]  # Close the polygon
        
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=model)
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    # Set y-axis limit
    ax.set_ylim(0, 1)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add title
    plt.title('Model Comparison Spider Chart', size=15, y=1.1)
    
    # Add note about queries per second
    plt.figtext(0.5, 0.01, "Queries/Second: Number of entries that can be processed in 1 second.", 
                ha='center', fontsize=10)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def load_glossary(glossary_file_path: str) -> str:
    """
    Load the glossary content from file.
    
    Args:
        glossary_file_path: Path to the glossary file
        
    Returns:
        String containing the glossary content
    """
    try:
        with open(glossary_file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "Glossary file not found."

def format_error_cell(error: Dict) -> str:
    """
    Format error information into a single cell with the specified format.
    
    Args:
        error: Dictionary containing error information
        
    Returns:
        Formatted string for the error cell
    """
    # Format the query line
    query_line = f"Qy: {error['query']}"
    query_line = truncate_text(query_line, 88)
    
    # Format the actual category/subcategory line
    if 'true_subcategory' in error:
        actual_line = f"AC: {error['true_category']} / {error['true_subcategory']} - {error['true_description']}"
    else:
        actual_line = f"AC: {error['true_category']} - {error['true_description']}"
    actual_line = truncate_text(actual_line, 88)
    
    # Format the first prediction line
    p1_sim_pct = error['similarity'] * 100
    p1_line = f"P1: {p1_sim_pct:.2f}% > {error['predicted_category']} / {error['predicted_subcategory']} - {error['predicted_description']}"
    p1_line = truncate_text(p1_line, 88)
    
    # Format the second prediction line
    p2_sim_pct = error['second_best_similarity'] * 100
    p2_line = f"P2: {p2_sim_pct:.2f}% > {error['second_best_category']} / {error['second_best_subcategory']} - {error['second_best_description']}"
    p2_line = truncate_text(p2_line, 88)
    
    # Combine all lines
    return f"{query_line}\n{actual_line}\n{p1_line}\n{p2_line}"

# Main execution code
def main():
    # Initialize logging and reporting
    logger, reporter = setup_logging_and_reporting(log_dir=RESULTS_DIR)
    
    # Add content to the report
    logger.add_to_report("heading1", "Query Router Model Comparison")
    
    logger.add_to_report("text", "This document presents a comparative analysis of different " 
                         "transformer models for query routing. We evaluate several models " 
                         "and present their performance metrics, confusion matrices, and " 
                         "incorrect classification examples.")

    # Load topics
    topics_df = load_topics("topics.yaml")
    
    logger.add_to_report("heading2", "Dataset and Topics")
    logger.add_to_report("text", f"Loaded {len(topics_df)} topics from {topics_df['category'].nunique()} categories")
    
    # Create full text descriptions for topics
    topic_texts = create_topic_texts(topics_df)
    
    # Process the clean dataset
    df_clean = load_csv_data("Lawn Care Dataset - Clear Categories with Occasional Typos.csv")
    logger.add_to_report("text", f"Loaded {len(df_clean)} queries from the dataset with occasional typos")
    
    # Create a table for model comparison
    model_comparison_table = [["Model", "Accuracy", "Precision", "Recall", "F1 Score", 
                             "Time (First 10)", "Time (Last 10)", "Queries/Second"]]
    
    # Dictionary to store metrics for spider chart
    model_metrics = {}
    
    # Dictionaries to store incorrect results for each model
    all_category_errors = {}
    all_subcategory_errors = {}
    
    # Process all models
    for model_name in models_to_test:
        logger.add_to_report("heading1", f"Model: {model_name}")
        logger.add_to_report("text", f"Evaluating model: {model_name}")
        
        # Initialize SentenceTransformer model for topic embeddings
        print(f"Initializing model: {model_name}")
        topic_model = SentenceTransformer(model_name)
        
        # Generate embeddings for topics
        topic_embeddings = topic_model.encode(topic_texts)
        logger.add_to_report("text", f"Generated embeddings with shape: {topic_embeddings.shape} for topics")
        
        # Get data from clean dataset
        queries = df_clean['query'].tolist()
        true_categories = df_clean['category'].tolist()
        true_subcategories = df_clean['subcategory'].tolist()
        
        # Classify queries with alternatives
        print(f"Classifying queries with model: {model_name}")
        results, time_first_10, time_last_10 = classify_queries_with_alternatives(
            queries, topic_embeddings, topics_df, model_name)
        
        # Calculate queries per second
        queries_per_second = 10 / time_last_10 if time_last_10 > 0 else 0
        
        # Get predictions
        predicted_categories = [result['predicted_category'] for result in results]
        
        # Get unique categories
        unique_categories = sorted(set(true_categories).union(set(predicted_categories)))
        
        # Create confusion matrix for categories
        cm_file_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name.replace('/', '-')}.png")
        print(f"Creating confusion matrix for model: {model_name}")
        create_confusion_matrix(true_categories, predicted_categories, unique_categories, cm_file_path)
        
        # Add confusion matrix image to report
        logger.add_to_report("heading2", "Confusion Matrix")
        logger.add_to_report("text", "The following confusion matrix shows the distribution of true vs predicted categories:")
        logger.add_to_report("image", cm_file_path, width=6)
        
        # Calculate metrics
        metrics = calculate_metrics(true_categories, predicted_categories, unique_categories)
        
        # Add metrics to report
        logger.add_to_report("heading2", "Performance Metrics")
        logger.add_to_report("text", "The following metrics summarize the model's performance:")
        
        metrics_table = [
            ["Metric", "Value"],
            ["Accuracy", f"{metrics['accuracy']:.4f}"],
            ["Precision", f"{metrics['precision']:.4f}"],
            ["Recall", f"{metrics['recall']:.4f}"],
            ["F1 Score", f"{metrics['f1_score']:.4f}"],
            ["Time to Classify First 10 Entries", f"{time_first_10:.4f} seconds"],
            ["Time to Classify Last 10 Entries", f"{time_last_10:.4f} seconds"],
            ["Queries per Second", f"{queries_per_second:.2f}"]
        ]
        
        logger.add_to_report("table", metrics_table)
        
        # Add to comparison table
        model_comparison_table.append([
            model_name,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1_score']:.4f}",
            f"{time_first_10:.4f}s",
            f"{time_last_10:.4f}s",
            f"{queries_per_second:.2f}"
        ])
        
        # Store metrics for spider chart
        # Normalize queries per second to a 0-1 scale
        # Assuming max 10 queries per second would be ideal
        max_queries_per_second = 10.0
        normalized_queries_per_second = min(queries_per_second / max_queries_per_second, 1.0)
        
        model_metrics[model_name] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'f1_score': metrics['f1_score'],
            'queries_per_second': normalized_queries_per_second
        }
        
        # Get category and subcategory errors
        category_errors = get_category_errors(results, true_categories, true_subcategories, topics_df)
        subcategory_errors = get_subcategory_errors(results, true_categories, true_subcategories, topics_df)
        
        all_category_errors[model_name] = category_errors
        all_subcategory_errors[model_name] = subcategory_errors
    
    # Create spider chart
    spider_chart_path = os.path.join(RESULTS_DIR, "model_comparison_spider_chart.png")
    create_spider_chart(model_metrics, spider_chart_path)
    
    # Add overall model comparison
    logger.add_to_report("heading1", "Model Comparison")
    logger.add_to_report("text", "The following table compares the performance of all tested models:")
    logger.add_to_report("table", model_comparison_table)
    
    # Add spider chart to report
    logger.add_to_report("text", "The following spider chart visualizes the performance of all models:")
    logger.add_to_report("image", spider_chart_path, width=6)
    logger.add_to_report("text", "Queries/Second: Number of entries that can be processed in 1 second.")
    
    # Add section for incorrect classifications
    logger.add_to_report("heading1", "Incorrect Classifications")
    
    for model_name in models_to_test:
        logger.add_to_report("heading2", f"Error reporting for {model_name}")
        
        # Category errors
        category_errors = all_category_errors[model_name]
        subcategory_errors = all_subcategory_errors[model_name]
        
        logger.add_to_report("heading3", "Category level errors")
        
        if not category_errors:
            logger.add_to_report("text", "No category-level errors for this model.")
        else:
            logger.add_to_report("text", f"Total category-level errors: {len(category_errors)}")
            
            # Create a table for category errors
            category_error_table = [["Error Details"]]
            
            for error in category_errors:
                formatted_error = format_error_cell(error)
                category_error_table.append([formatted_error])
            
            logger.add_to_report("table", category_error_table)
        
        # Subcategory errors
        logger.add_to_report("heading3", "Subcategory level errors")
        
        if not subcategory_errors:
            logger.add_to_report("text", "No subcategory-level errors for this model.")
        else:
            logger.add_to_report("text", f"Total subcategory-level errors: {len(subcategory_errors)}")
            
            # Create a table for subcategory errors
            subcategory_error_table = [["Error Details"]]
            
            for error in subcategory_errors:
                formatted_error = format_error_cell(error)
                subcategory_error_table.append([formatted_error])
            
            logger.add_to_report("table", subcategory_error_table)
    
    # Add glossary section
    logger.add_to_report("heading1", "Annexes")
    logger.add_to_report("heading2", "Glossary")
    
    # Load and add glossary content
    glossary_content = load_glossary(GLOSSARY_FILE)
    logger.add_to_report("text", glossary_content)
    
    # Generate the report
    report_path = reporter.generate_report(
        title="\n\n\nQuery Router Model Comparison",
        subtitle="\n\n\nPerformance Analysis of Various Transformer Models",
        author="\n\n\n\n\n\n\n\n\n\n\n\n\nNLP/ML Expert Gilles Ferrero\n\ngilles.ferrero@gmail.com",
        organization="\n\nAudacity Ltd.",
        intro_file=INTRO_FILE
    )
    
    print(f"\nExperiment completed successfully.")
    print(f"Report generated at: {report_path}")

if __name__ == "__main__":
    main()