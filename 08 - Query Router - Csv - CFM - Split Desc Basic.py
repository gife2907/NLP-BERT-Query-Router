import yaml
import pandas as pd
import numpy as np
import time
import os
import logging
import datetime
import glob
import traceback
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from typing import List, Dict, Tuple

# Import the enhanced reporting library
from Libraries.reporter_lib import (
    setup_logging_and_reporting,
    create_numbered_results_dir,
    setup_base_logging,
    create_confusion_matrix,
    calculate_metrics,
    export_results_to_excel,
    get_category_errors,
    get_subcategory_errors,
    get_correct_classifications,
    find_close_categories
)

# Constants
LIBRARIES_DIR = "Libraries"
INTRO_FILE = os.path.join(LIBRARIES_DIR, "intro_about_me.txt")
GLOSSARY_FILE = os.path.join(LIBRARIES_DIR, "classification_glossary.txt")

# Setup logging and reporting
logger, reporter, excel_reporter, RESULTS_DIR, base_logger = setup_logging_and_reporting()

# Define models to test
models_to_test = [
    'bert-base-uncased',
    'prajjwal1/bert-mini',
    'all-MiniLM-L6-v2',
   # 'microsoft/deberta-v3-base',
   # 'sentence-transformers/all-mpnet-base-v2',
   # 'sentence-transformers/all-roberta-large-v1'
]

def load_csv_data(csv_file_path: str) -> pd.DataFrame:
    """
    Load test data from CSV file with structure: query,category,subcategory
    
    Args:
        csv_file_path: Path to the CSV file containing test data
        
    Returns:
        DataFrame with columns: query, category, subcategory
    """
    base_logger.info(f"Loading CSV data from {csv_file_path}")
    return pd.read_csv(csv_file_path, skiprows=1, names=['query', 'category', 'subcategory'])

def load_topics(topics_file_path: str) -> pd.DataFrame:
    """
    Load topics from the YAML file into a DataFrame structure.
    
    Args:
        topics_file_path: Path to the topics YAML file
        
    Returns:
        DataFrame with topic classification, title, and description
    """
    base_logger.info(f"Loading topics from {topics_file_path}")
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
    topics_df = pd.DataFrame(all_topics)
    base_logger.info(f"Loaded {len(topics_df)} topics")
    return topics_df

def create_topic_texts(topics_df: pd.DataFrame) -> List[str]:
    """
    Create full text descriptions for topics to generate embeddings.
    
    Args:
        topics_df: DataFrame with topic information
        
    Returns:
        List of combined topic texts
    """
    base_logger.info("Creating topic texts for embeddings")
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
    base_logger.info(f"Classifying queries using model: {model_name}")
    
    # Initialize model
    model = SentenceTransformer(model_name)
    
    results = []
    
    # Measure time for first 10 entries
    base_logger.info("Processing first 10 queries")
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
        
        # Calculate distance between the topic vectors themselves
        topic_vectors_similarity = cosine_similarity([topic_embeddings[top_indices[0]]], [topic_embeddings[top_indices[1]]])[0][0]
        
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
            'cosine_distance': cosine_distance,
            'topic_vectors_similarity': topic_vectors_similarity
        })
    
    time_for_first_10 = time.time() - start_time_first
    base_logger.info(f"Time for first 10 queries: {time_for_first_10:.2f} seconds")
    
    # Process the rest of the queries (excluding the first 10)
    base_logger.info("Processing middle queries")
    middle_queries = queries[10:-10]
    if middle_queries:
        middle_embeddings = model.encode(middle_queries)
        
        for i, query in enumerate(middle_queries):
            similarities = cosine_similarity([middle_embeddings[i]], topic_embeddings)[0]
            top_indices = np.argsort(similarities)[-2:][::-1]
            
            predicted_topic = topics_df.iloc[top_indices[0]]
            second_best_topic = topics_df.iloc[top_indices[1]]
            cosine_distance = similarities[top_indices[0]] - similarities[top_indices[1]]
            topic_vectors_similarity = cosine_similarity([topic_embeddings[top_indices[0]]], [topic_embeddings[top_indices[1]]])[0][0]
            
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
                'cosine_distance': cosine_distance,
                'topic_vectors_similarity': topic_vectors_similarity
            })
    
    # Measure time for last 10 entries
    base_logger.info("Processing last 10 queries")
    start_time_last = time.time()
    last_10_queries = queries[-10:]
    last_10_embeddings = model.encode(last_10_queries)
    
    for i, query in enumerate(last_10_queries):
        similarities = cosine_similarity([last_10_embeddings[i]], topic_embeddings)[0]
        top_indices = np.argsort(similarities)[-2:][::-1]
        
        predicted_topic = topics_df.iloc[top_indices[0]]
        second_best_topic = topics_df.iloc[top_indices[1]]
        cosine_distance = similarities[top_indices[0]] - similarities[top_indices[1]]
        topic_vectors_similarity = cosine_similarity([topic_embeddings[top_indices[0]]], [topic_embeddings[top_indices[1]]])[0][0]
        
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
            'cosine_distance': cosine_distance,
            'topic_vectors_similarity': topic_vectors_similarity
        })
    
    time_for_last_10 = time.time() - start_time_last
    base_logger.info(f"Time for last 10 queries: {time_for_last_10:.2f} seconds")
    
    base_logger.info(f"Classified {len(results)} queries in total")
    return results, time_for_first_10, time_for_last_10

def load_glossary(glossary_file_path: str) -> str:
    """
    Load the glossary content from file.
    
    Args:
        glossary_file_path: Path to the glossary file
        
    Returns:
        String containing the glossary content
    """
    try:
        base_logger.info(f"Loading glossary from {glossary_file_path}")
        with open(glossary_file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        base_logger.warning(f"Glossary file not found: {glossary_file_path}")
        return "Glossary file not found."

def main():
    try:
        # Start time for overall processing
        start_time = time.time()
        base_logger.info("Starting query router analysis")

        # Log versions and parameters used
        base_logger.info(f"Models to test: {models_to_test}")
        
        # Add introduction to the report
        if os.path.exists(INTRO_FILE):
            with open(INTRO_FILE, 'r') as file:
                intro_text = file.read()
            logger.add_to_report("heading1", "Introduction")
            logger.add_to_report("text", intro_text)
        
        # Save glossary content to add at the end in appendix
        glossary_text = load_glossary(GLOSSARY_FILE)
        
        # Load topics configuration from YAML
        topics_file = "topics.yaml"  # Placeholder - replace with actual path
        topics_df = load_topics(topics_file)
        logger.add_to_report("heading2", "Topic Classification Structure")
        logger.add_to_report("text", f"Loaded {len(topics_df)} topics from {topics_file}")
        
        # Create topic texts for embeddings
        topic_texts = create_topic_texts(topics_df)
        
        # Store results per model
        model_results = {}
        
        # Process each model
        for model_name in models_to_test:
            base_logger.info(f"Processing model: {model_name}")
            logger.add_to_report("heading1", f"Model: {model_name}")
            
            # Load model and generate topic embeddings
            base_logger.info(f"Loading model: {model_name}")
            model = SentenceTransformer(model_name)
            
            base_logger.info("Generating topic embeddings")
            topic_embeddings = model.encode(topic_texts)

            # Find close categories with the updated 90% threshold
            base_logger.info(f"Finding close categories with 90% threshold for {model_name}")
            close_categories_df, threshold_message = find_close_categories(topics_df, topic_embeddings, threshold=0.90)

            # Add the threshold information to the report
            logger.add_to_report("heading2", "Close Categories Analysis")
            logger.add_to_report("text", threshold_message)

            if not close_categories_df.empty:
                logger.add_to_report("text", f"Found {len(close_categories_df)} pairs of closely related categories.")
                
                # Create a more readable summary for the report using the new formatted output
                logger.add_to_report("heading3", "Detailed Category Similarities")
                
                # Add each formatted pair to the report
                for _, row in close_categories_df.iterrows():
                    logger.add_to_report("text", row['Formatted_Output'])
                    
                # Optional: Save the close categories to a separate file for reference
                close_categories_file = os.path.join(RESULTS_DIR, f"{model_name.replace('/', '-')}_close_categories.csv")
                close_categories_df.to_csv(close_categories_file, index=False)
                logger.add_to_report("text", f"\nDetailed close category analysis saved to: {close_categories_file}")
            else:
                logger.add_to_report("text", "No closely related categories found with the current threshold.")
                logger.add_to_report("text", "Consider lowering the threshold if you expect to find similar categories.")            
                
            # Load and classify test data
            csv_test_file = "Lawn Care Dataset - Clear Categories with Occasional Typos.csv"  # Placeholder - replace with actual path
            test_data_df = load_csv_data(csv_test_file)
            
            # Get queries, categories, and subcategories from test data
            queries = test_data_df['query'].tolist()
            true_categories = test_data_df['category'].tolist()
            true_subcategories = test_data_df['subcategory'].tolist()
            
            logger.add_to_report("heading2", "Test Data")
            logger.add_to_report("text", f"Loaded {len(queries)} test queries from {csv_test_file}")
            
            # Classify queries
            base_logger.info(f"Classifying queries with model: {model_name}")
            results, time_first_10, time_last_10 = classify_queries_with_alternatives(
                queries, topic_embeddings, topics_df, model_name
            )
            
            # Calculate average time per query
            total_time = time_first_10 + time_last_10
            avg_time = (time_first_10 + time_last_10) / 20 if len(queries) >= 20 else total_time / len(queries)
            queries_per_second = 1 / avg_time
            
            logger.add_to_report("heading2", "Classification Results")
            logger.add_to_report("text", f"Time for first 10 queries: {time_first_10:.4f} seconds")
            logger.add_to_report("text", f"Time for last 10 queries: {time_last_10:.4f} seconds")
            logger.add_to_report("text", f"Average time per query: {avg_time:.4f} seconds")
            logger.add_to_report("text", f"Queries per second: {queries_per_second:.2f}")
            
            # Get predicted categories and subcategories
            predicted_categories = [result['predicted_category'] for result in results]
            predicted_subcategories = [result['predicted_subcategory'] for result in results]
            
            # Calculate accuracy for categories
            category_accuracy = accuracy_score(true_categories, predicted_categories)
            logger.add_to_report("text", f"Category accuracy: {category_accuracy * 100:.2f}%")
            
            # Calculate accuracy for category+subcategory combinations
            true_combinations = [f"{cat}::{subcat}" for cat, subcat in zip(true_categories, true_subcategories)]
            predicted_combinations = [f"{cat}::{subcat}" for cat, subcat in zip(predicted_categories, predicted_subcategories)]
            combination_accuracy = accuracy_score(true_combinations, predicted_combinations)
            logger.add_to_report("text", f"Category+Subcategory accuracy: {combination_accuracy * 100:.2f}%")
            
            # Calculate metrics for category level
            category_metrics = calculate_metrics(true_categories, predicted_categories, list(set(true_categories)))
            
            # Calculate metrics for category+subcategory combinations
            combo_metrics = calculate_metrics(true_combinations, predicted_combinations, list(set(true_combinations)))
            
            # Store metrics for model comparison
            model_results[model_name] = {
                'accuracy': combo_metrics['accuracy'],
                'precision': combo_metrics['precision'],
                'recall': combo_metrics['recall'],
                'f1_score': combo_metrics['f1_score'],
                'queries_per_second': queries_per_second,
                'time_per_query': avg_time,
                'results': results,
                'true_categories': true_categories,
                'category_accuracy': category_accuracy,
                'category_precision': category_metrics['precision'],
                'category_recall': category_metrics['recall'],
                'category_f1': category_metrics['f1_score']
            }
            
            # Add confusion matrices
            base_logger.info("Creating confusion matrices")
            
            # For categories
            unique_categories = sorted(list(set(true_categories + predicted_categories)))
            category_cm_file = os.path.join(RESULTS_DIR, f"{model_name.replace('/', '-')}_category_cm.png")
            create_confusion_matrix(true_categories, predicted_categories, unique_categories, category_cm_file)
            logger.add_to_report("heading3", "Category Confusion Matrix")
            logger.add_to_report("image", category_cm_file)
            
            # For subcategories
            unique_subcategories = sorted(list(set(true_subcategories + predicted_subcategories)))
            subcat_cm_file = os.path.join(RESULTS_DIR, f"{model_name.replace('/', '-')}_subcategory_cm.png")
            create_confusion_matrix(true_subcategories, predicted_subcategories, unique_subcategories, subcat_cm_file)
            logger.add_to_report("heading3", "Subcategory Confusion Matrix")
            logger.add_to_report("image", subcat_cm_file)
            
            # Analyze errors
            base_logger.info("Analyzing errors")
            
            # Get category errors
            category_errors = get_category_errors(results, true_categories, true_subcategories, topics_df)
            logger.add_to_report("heading3", "Category Errors")
            logger.add_to_report("text", f"Number of category errors: {len(category_errors)}")
            
            # Get subcategory errors
            subcategory_errors = get_subcategory_errors(results, true_categories, true_subcategories, topics_df)
            logger.add_to_report("heading3", "Subcategory Errors")
            logger.add_to_report("text", f"Number of subcategory errors: {len(subcategory_errors)}")
            
            # Get correct classifications
            correct_items = get_correct_classifications(results, true_categories, true_subcategories, topics_df)
            logger.add_to_report("heading3", "Correct Classifications")
            logger.add_to_report("text", f"Number of correct classifications: {len(correct_items)}")
            
            # Export results to Excel for detailed analysis
            base_logger.info("Exporting results to Excel")
            excel_file = export_results_to_excel(
                category_errors, subcategory_errors, correct_items, model_name, RESULTS_DIR
            )
            logger.add_to_report("text", f"Detailed results exported to: {excel_file}")
        
        # Add Category-level performance comparison
        base_logger.info("Creating category-level performance comparison")
        logger.add_to_report("heading1", "Category-Level Model Comparison")
        logger.add_to_report("text", "This comparison shows model performance at the CATEGORY level only (ignoring subcategories):")
        
        # Create category-level comparison table
        category_comparison_data = []
        for model_name in model_results:
            category_comparison_data.append([
                model_name,
                f"{model_results[model_name]['category_accuracy']*100:.2f}%",
                f"{model_results[model_name]['category_precision']*100:.2f}%",
                f"{model_results[model_name]['category_recall']*100:.2f}%",
                f"{model_results[model_name]['category_f1']*100:.2f}%",
                f"{model_results[model_name]['queries_per_second']:.2f}",
                f"{model_results[model_name]['time_per_query']*1000:.2f} ms"
            ])
        
        # Define the headers
        category_comparison_headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Queries/sec", "Time/query"]
        
        # Add to report
        logger.add_to_report("table", [category_comparison_headers] + category_comparison_data)
        
        # Overall subcategory-level performance comparison
        base_logger.info("Creating subcategory-level performance comparison")
        logger.add_to_report("heading1", "Category+Subcategory-Level Model Comparison")
        logger.add_to_report("text", "This comparison shows model performance at the CATEGORY+SUBCATEGORY level (both must be correct):")
        
        # Create comparison table
        comparison_data = []
        for model_name, metrics in model_results.items():
            comparison_data.append([
                model_name,
                f"{metrics['accuracy']*100:.2f}%",
                f"{metrics['precision']*100:.2f}%",
                f"{metrics['recall']*100:.2f}%",
                f"{metrics['f1_score']*100:.2f}%",
                f"{metrics['queries_per_second']:.2f}",
                f"{metrics['time_per_query']*1000:.2f} ms"
            ])
        
        comparison_headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Queries/sec", "Time/query"]
        logger.add_to_report("table", [comparison_headers] + comparison_data)
        
        # Overall execution time
        execution_time = time.time() - start_time
        logger.add_to_report("heading1", "Execution Summary")
        logger.add_to_report("text", f"Total execution time: {execution_time:.2f} seconds")
        
        # Add glossary at the end in appendix section
        if glossary_text != "Glossary file not found.":
            logger.add_to_report("heading1", "Appendix")
            logger.add_to_report("heading2", "Classification Glossary")
            logger.add_to_report("text", glossary_text)
        
        # Generate final report
        report_file = "query_router_report.docx"
        report = reporter.generate_report(
            output_filename=report_file,
            title="Query Router Analysis Report",
            subtitle="Model Comparison and Classification Results",
            intro_file=None  # Set to None to avoid double introduction
        )
        base_logger.info(f"Report generated: {report}")
        
    except Exception as e:
        base_logger.error(f"Error in query router analysis: {str(e)}")
        base_logger.error(traceback.format_exc())
        logger.add_to_report("heading1", "Error")
        logger.add_to_report("text", f"An error occurred during analysis: {str(e)}")
        logger.add_to_report("text", traceback.format_exc())

if __name__ == "__main__":
    main()