import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import json
import os

class JobSelector:
    def __init__(self, file_path, method="SAW"):
        self.file_path = file_path
        self.method = method
        self.decision_matrix = None
        self.weights = None
        self.criteria = None
        self.criteria_type = None

    def parse_excel(self):
        excel_data = pd.ExcelFile(self.file_path)
        alternatives_criteria = excel_data.parse('Alternatives-Criteria')
        self.decision_matrix = alternatives_criteria.drop('Alternative', axis=1)
        self.decision_matrix.index = alternatives_criteria['Alternative']
        weights_df = excel_data.parse('Weights')
        self.weights = np.array(weights_df['Weight'])
        self.criteria = list(weights_df['Criterion'])
        self.criteria_type = list(weights_df['Type'])

    def parse_json(self):
        with open(self.file_path, 'r') as file:
            data = json.load(file)
        self.decision_matrix = pd.DataFrame(data['data'], index=data['index'], columns=data['columns'])
        self.weights = np.array(data['weights'])
        self.criteria = data['columns']
        self.criteria_type = data['criteria_type']

    def normalize_matrix(self):
        normalized_matrix = self.decision_matrix.astype('float64')
        for i, criterion_type in enumerate(self.criteria_type):
            if not pd.api.types.is_numeric_dtype(normalized_matrix.iloc[:, i]):
                raise ValueError(f"Column '{self.decision_matrix.columns[i]}' contains non-numeric values.")
            if criterion_type == 'Benefit':
                normalized_matrix.iloc[:, i] = normalized_matrix.iloc[:, i] / normalized_matrix.iloc[:, i].max()
            elif criterion_type == 'Cost':
                normalized_matrix.iloc[:, i] = normalized_matrix.iloc[:, i].min() / normalized_matrix.iloc[:, i]
        return normalized_matrix

    def apply_saw_method(self, normalized_matrix):
        scores = np.dot(self.weights, normalized_matrix.T)
        return scores
            
    def plot_results(self, scores):
        scores_df = pd.DataFrame(scores, index=self.decision_matrix.index, columns=["Score"])
        scores_df = scores_df.sort_values(by="Score", ascending=False)
        ax = scores_df["Score"].plot(kind="bar", colormap="viridis", xlabel="Alternatives", ylabel="Score", rot=0)
        plt.title("Job Alternatives - Scores")
        plt.savefig('./out/job_scores.png', bbox_inches='tight')
        return scores_df

    def run(self):
        file_extension = os.path.splitext(self.file_path)[1].lower()

        if file_extension == '.json':
            print("\nLoading JSON data...")
            self.parse_json()
        elif file_extension == '.xlsx':
            print("\nLoading Excel data...")
            self.parse_excel()
        else:
            raise ValueError("\nUnsupported file type. Please provide a .json or .xlsx file.")
        
        print("Data successfully loaded.")

        normalized_matrix = self.normalize_matrix()

        if self.method == "SAW":
            print("\nUsing SAW method for scoring...")
            scores = self.apply_saw_method(normalized_matrix)
        else:
            raise ValueError(f"\nAlgorithm '{self.method}' is not implemented.")

        final_scores = self.plot_results(scores)

        print("\nFinal rankings:")
        print(final_scores)

        top_choice = final_scores.index[0]
        print(f"\n...You should consider selecting {top_choice}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Attribute Decision Making for Job Selection")
    parser.add_argument("--file", type=str, required=True, help="Path to the input file (.json or .xlsx)")
    parser.add_argument("--method", type=str, default="SAW", choices=["SAW"], help="Method to use for scoring")
    args = parser.parse_args()

    job_selector = JobSelector(file_path=args.file, method=args.method)
    job_selector.run()
