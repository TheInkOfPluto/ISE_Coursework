import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy import stats
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# ========== Global Configuration ==========
REPEAT = 10
TFIDF_MAX_FEATURES = 1000
TFIDF_NGRAM_RANGE = (1, 2)
STOP_WORDS = stopwords.words('english') + ['...']

MODEL_CONFIGS = {
    'GaussianNB': (GaussianNB(), {'var_smoothing': np.logspace(-12, 0, 13)}),
    'MultinomialNB': (MultinomialNB(), {'alpha': np.logspace(-3, 1, 5)}),
    'LogisticRegression': (LogisticRegression(max_iter=1000), {'C': np.logspace(-2, 2, 5)}),
    'RandomForest': (RandomForestClassifier(), {'n_estimators': [50, 100], 'max_depth': [5, 10]}),
    'LinearSVC': (LinearSVC(max_iter=2000), {'C': np.logspace(-2, 2, 5)}),
    'SGDClassifier': (SGDClassifier(max_iter=1000), {'alpha': np.logspace(-4, -1, 4)}),
    'KNeighborsClassifier': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {'n_estimators': [50, 100]}),
    'LightGBM': (LGBMClassifier(), {'n_estimators': [50, 100]})
}

# ========== Preprocessing ==========
def remove_html(text):
    return re.sub(r'<.*?>', '', str(text))

def remove_emoji(text):
    emoji_pattern = re.compile(
        "[" +
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\u2702-\u27B0"
        u"\u24C2-\U0001F251" +
        "]", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', str(text))

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word.lower() not in STOP_WORDS])

def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def preprocess_text(text):
    text = remove_html(text)
    text = remove_emoji(text)
    text = remove_stopwords(text)
    return clean_str(text)

# ========== Core Training ==========
def run_model_experiment(model_name, model, param_grid, X, y, output_dir, project_name):
    accuracies, precisions, recalls, f1_scores, auc_values = [], [], [], [], []

    for seed in range(REPEAT):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_clf = grid.best_estimator_
        y_pred = best_clf.predict(X_test)

        if hasattr(best_clf, "predict_proba"):
            y_prob = best_clf.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else np.zeros_like(y_pred)
        else:
            y_prob = np.zeros_like(y_pred)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        recalls.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))

        try:
            fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
            auc_val = auc(fpr, tpr)
        except:
            auc_val = float('nan')
        auc_values.append(auc_val)

    metrics = {
        'Model': model_name,
        'Accuracy': np.mean(accuracies),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'F1': np.mean(f1_scores),
        'AUC': np.mean(auc_values)
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df = pd.DataFrame([metrics])
    df.to_csv(os.path.join(output_dir, f'{project_name}_{model_name}_{timestamp}.csv'), index=False)

    return metrics

# ========== Project Runner ==========
def run_all_projects(data_dir='datasets', output_dir='9WAYS_final'):
    os.makedirs(output_dir, exist_ok=True)
    all_projects_metrics = []
    all_f1_scores = {}

    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            project_path = os.path.join(data_dir, file)
            project_name = file.replace('.csv', '')
            print(f"\n🚀 Running project: {project_name}")

            df = pd.read_csv(project_path).sample(frac=1, random_state=999)
            df['text'] = df.apply(
                lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'], axis=1)
            df = df.rename(columns={"Unnamed: 0": "id", "class": "sentiment"})
            df['text'] = df['text'].fillna('').apply(preprocess_text)

            tfidf = TfidfVectorizer(ngram_range=TFIDF_NGRAM_RANGE, max_features=TFIDF_MAX_FEATURES)
            X = tfidf.fit_transform(df['text']).toarray()
            y = df['sentiment']

            project_metrics = []
            for model_name, (model, param_grid) in MODEL_CONFIGS.items():
                metrics = run_model_experiment(model_name, model, param_grid, X, y, output_dir, project_name)
                metrics['Project'] = project_name
                all_projects_metrics.append(metrics)
                project_metrics.append(metrics)
                all_f1_scores.setdefault(model_name, []).append(metrics['F1'])

            compare_df = pd.DataFrame(project_metrics)
            compare_df.set_index('Model')[['Precision', 'Recall', 'F1']].plot(kind='bar', figsize=(10, 5))
            plt.ylim(0, 1)
            plt.title(f'{project_name} - Model Comparison')
            plt.ylabel('Score')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{project_name}_comparison.png"))
            plt.close()

            compare_df.to_csv(os.path.join(output_dir, f"{project_name}_all_models_summary.csv"), index=False)

            fig, ax = plt.subplots(figsize=(12, 3))
            ax.axis('off')
            tbl_data = [compare_df.columns.tolist()] + compare_df.values.tolist()
            table = ax.table(cellText=tbl_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            plt.savefig(os.path.join(output_dir, f"{project_name}_all_models_summary.png"), bbox_inches='tight')
            plt.close()

    summary_df = pd.DataFrame(all_projects_metrics)
    avg_df = summary_df.groupby('Model').mean(numeric_only=True).reset_index()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    avg_df.to_csv(os.path.join(output_dir, f"ALL_PROJECTS_summary_{timestamp}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    tbl_data = [avg_df.columns.tolist()] + avg_df.values.tolist()
    table = ax.table(cellText=tbl_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    plt.savefig(os.path.join(output_dir, f"ALL_PROJECTS_summary_{timestamp}.png"), bbox_inches='tight')
    plt.close()

    print(f"\n✅ ALL PROJECTS summary saved.")

    # ========== Statistical Significance Testing ==========
    print("\n📊 Running statistical significance tests...")

    model_names = list(all_f1_scores.keys())
    f1_matrix = [all_f1_scores[name] for name in model_names]

    markdown_lines = []
    markdown_lines.append(f"## Statistical Significance Results ({timestamp})\n")

    f_stat, p_val = stats.f_oneway(*f1_matrix)
    markdown_lines.append(f"**ANOVA F-statistic**: {f_stat:.4f}  ")
    markdown_lines.append(f"**p-value**: {p_val:.4f}\n")

    if p_val < 0.05:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        import warnings
        warnings.filterwarnings("ignore")

        all_scores = sum(f1_matrix, [])
        model_labels = sum([[name] * len(scores) for name, scores in zip(model_names, f1_matrix)], [])
        tukey = pairwise_tukeyhsd(endog=all_scores, groups=model_labels, alpha=0.05)
        markdown_lines.append("\n### Tukey HSD Post-hoc Test\n")
        markdown_lines.append("| Group1 | Group2 | Mean Diff | p-adj | Lower | Upper | Reject |")
        markdown_lines.append("|--------|--------|-----------|-------|-------|-------|--------|")
        for row in tukey.summary().data[1:]:
            markdown_lines.append("| " + " | ".join(str(col) for col in row) + " |")

    baseline_scores = all_f1_scores['GaussianNB']
    markdown_lines.append("\n### Paired t-tests vs Baseline (GaussianNB)\n")
    markdown_lines.append("| Model | t-statistic | p-value | Significant (p<0.05) |")
    markdown_lines.append("|--------|-------------|---------|----------------------|")
    for name in model_names:
        if name != 'GaussianNB':
            t_stat, t_p = stats.ttest_rel(all_f1_scores[name], baseline_scores)
            significance = "✅" if t_p < 0.05 else "❌"
            markdown_lines.append(f"| {name} | {t_stat:.4f} | {t_p:.4f} | {significance} |")

    markdown_lines.append("\n*All tests based on [SciPy stats module](https://docs.scipy.org/doc/scipy/reference/stats.html)*\n")

    with open(os.path.join(output_dir, f"ALL_PROJECTS_significance_{timestamp}.md"), 'w', encoding='utf-8') as f:
        f.write("\n".join(markdown_lines))

    print("\n✅ Statistical tests complete. Markdown saved.")

# ========== Entry ==========
if __name__ == '__main__':
    run_all_projects()
