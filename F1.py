import polars as pl

def main(pred_path: str, truth_path: str, output_path: str) -> None:
    pred_df = pl.read_csv(pred_path).select(['article_id', 'dataset_id', 'type'])
    truth_df = pl.read_csv(truth_path).select(['article_id', 'dataset_id', 'type']).filter(pl.col('type') != 'Missing')

    # Concatenate truth and pred with source indicator
    all_df = pl.concat([
        truth_df.with_columns(pl.lit('truth').alias('source')),
        pred_df.with_columns(pl.lit('pred').alias('source'))
    ])

    # Group for cat_type: per (article_id, dataset_id, type)
    type_grouped = all_df.group_by(['article_id', 'dataset_id', 'type']).agg(
        pl.col('source').unique().alias('sources')
    ).with_columns(
        pl.col('sources').list.contains('truth').alias('is_truth'),
        pl.col('sources').list.contains('pred').alias('is_pred')
    ).with_columns(
        pl.when(pl.col('is_truth') & pl.col('is_pred')).then(pl.lit('TP'))
          .when(pl.col('is_truth')).then(pl.lit('FN'))
          .when(pl.col('is_pred')).then(pl.lit('FP'))
          .alias('cat_type'),
        pl.when(pl.col('is_truth')).then(pl.col('type')).otherwise(None).alias('type_true'),
        pl.when(pl.col('is_pred')).then(pl.col('type')).otherwise(None).alias('type_pred')
    ).drop(['type', 'sources', 'is_truth', 'is_pred'])

    # Group for cat_id: per (article_id, dataset_id)
    id_grouped = all_df.select(['article_id', 'dataset_id', 'source']).unique().group_by(['article_id', 'dataset_id']).agg(
        pl.col('source').unique().alias('sources')
    ).with_columns(
        pl.col('sources').list.contains('truth').alias('is_truth'),
        pl.col('sources').list.contains('pred').alias('is_pred')
    ).with_columns(
        pl.when(pl.col('is_truth') & pl.col('is_pred')).then(pl.lit('TP'))
          .when(pl.col('is_truth')).then(pl.lit('FN'))
          .when(pl.col('is_pred')).then(pl.lit('FP'))
          .alias('cat_id')
    ).drop(['sources', 'is_truth', 'is_pred'])

    # Group for article_id: per (article_id)
    article_id_grouped = all_df.select(['article_id', 'source']).unique().group_by(['article_id']).agg(
        pl.col('source').unique().alias('sources')
    ).with_columns(
        pl.col('sources').list.contains('truth').alias('is_truth'),
        pl.col('sources').list.contains('pred').alias('is_pred')
    ).with_columns(
        pl.when(pl.col('is_truth') & pl.col('is_pred')).then(pl.lit('TP'))
          .when(pl.col('is_truth')).then(pl.lit('FN'))
          .when(pl.col('is_pred')).then(pl.lit('FP'))
          .alias('cat_article_id')
    ).drop(['sources', 'is_truth', 'is_pred'])

    # Join cat_id to the type-level DF
    final_df = type_grouped.join(id_grouped, on=['article_id', 'dataset_id'], how='left')
    
    # Join cat_article_id to the final DF
    final_df = final_df.join(article_id_grouped, on=['article_id'], how='left')
    
    final_df = final_df.select(['article_id', 'dataset_id', 'type_true', 'type_pred', 'cat_id', 'cat_type', 'cat_article_id'])

    # Write to output
    final_df.write_csv(output_path)

if __name__ == '__main__':
    pred_path = './temp/submission.csv'
    truth_path = './data/train_labels.csv'  # Assume this is the path to the ground truth CSV; adjust as needed
    output_path = './temp/F1_details.csv'
    main(pred_path, truth_path, output_path)