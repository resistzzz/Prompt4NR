import pyarrow.parquet as pa

if __name__ == "__main__":
    # Read in dataset

    # BEHAVIORS
    train_behaviors_path = "DATA/ebnerd/train/behaviors.parquet"
    val_behaviors_path = "DATA/ebnerd/validation/behaviors.parquet"

    train_behaviors = pa.read_table(train_behaviors_path) 
    val_behaviors = pa.read_table(val_behaviors_path)

    # HISTORY
    train_history_path = "DATA/ebnerd/train/history.parquet"
    val_history_path = "DATA/ebnerd/validation/history.parquet"

    train_history = pa.read_table(train_history_path) 
    val_history = pa.read_table(val_history_path)

    # ARTICLES
    articles_path = "DATA/ebnerd/articles.parquet"

    articles = pa.read_table(articles_path)

    # * Processing behaviors
    # By default non-existent features in test set are: Article ID, Next Readtime, Next Scroll Percentage, and Clicked Article IDs.
    # Dropppinng where article_id == empty

