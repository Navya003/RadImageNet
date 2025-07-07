# Function to get user input with default "No response provided"
def get_user_input(prompt):
    response = input(prompt + " (Provide a full answer or press Enter for default): ")
    return response.strip() if response.strip() else "No response provided"

# Define the sections and questions
sections = {
    "TITLE / ABSTRACT": [
        "Identification as a study of AI methodology, specifying the category of technology used (e.g., deep learning)",
        "Structured summary of study design, methods, results, and conclusions"
    ],
    "INTRODUCTION": [
        "Scientific and clinical background, including the intended use and clinical role of the AI approach",
        "Study objectives and hypotheses"
    ],
    "METHODS": {
        "Study Design": [
            "Prospective or retrospective study",
            "Study goal, such as model creation, exploratory study, feasibility study, non-inferiority trial"
        ],
        "Data": [
            "Data sources",
            "Eligibility criteria: how, where, and when potentially eligible participants or studies were identified (e.g., symptoms, results from previous tests, inclusion in registry, patient-care setting, location, dates)",
            "Data pre-processing steps",
            "Selection of data subsets, if applicable",
            "Definitions of data elements, with references to Common Data Elements",
            "De-identification methods",
            "How missing data were handled"
        ],
        "Ground Truth": [
            "Definition of ground truth reference standard, in sufficient detail to allow replication",
            "Rationale for choosing the reference standard (if alternatives exist)",
            "Source of ground-truth annotations; qualifications and preparation of annotators",
            "Annotation tools",
            "Measurement of inter- and intrarater variability; methods to mitigate variability and/or resolve discrepancies"
        ],
        "Data Partitions": [
            "Intended sample size and how it was determined",
            "How data were assigned to partitions; specify proportions",
            "Level at which partitions are disjoint (e.g., image, study, patient, institution)"
        ],
        "Model": [
            "Detailed description of model, including inputs, outputs, all intermediate layers and connections",
            "Software libraries, frameworks, and packages",
            "Initialization of model parameters (e.g., randomization, transfer learning)"
        ],
        "Training": [
            "Details of training approach, including data augmentation, hyperparameters, number of models trained",
            "Method of selecting the final model",
            "Ensembling techniques, if applicable"
        ],
        "Evaluation": [
            "Metrics of model performance",
            "Statistical measures of significance and uncertainty (e.g., confidence intervals)",
            "Robustness or sensitivity analysis",
            "Methods for explainability or interpretability (e.g., saliency maps), and how they were validated",
            "Validation or testing on external data"
        ]
    },
    "RESULTS": {
        "Data": [
            "Flow of participants or cases, using a diagram to indicate inclusion and exclusion",
            "Demographic and clinical characteristics of cases in each partition"
        ],
        "Model performance": [
            "Performance metrics for optimal model(s) on all data partitions",
            "Estimates of diagnostic accuracy and their precision (such as 95% confidence intervals)",
            "Failure analysis of incorrectly classified cases"
        ]
    },
    "DISCUSSION": [
        "Study limitations, including potential bias, statistical uncertainty, and generalizability",
        "Implications for practice, including the intended use and/or clinical role"
    ],
    "OTHER INFORMATION": [
        "Registration number and name of registry",
        "Where the full study protocol can be accessed",
        "Sources of funding and other support; role of funders"
    ]
}

# Get responses from the user
responses = []
question_number = 1
for section, content in sections.items():
    if isinstance(content, dict):
        for subsection, questions in content.items():
            for question in questions:
                response = get_user_input(f"{question_number}. {question}")
                responses.append((question_number, question, response, section, subsection))
                question_number += 1
    else:
        for question in content:
            response = get_user_input(f"{question_number}. {question}")
            responses.append((question_number, question, response, section, ""))  # No subsection
            question_number += 1

# Function to create HTML report with CSS styling
def create_html(filename, responses):
    # Prompt user for author name and affiliation
    author_name = get_user_input("Author's Name:")
    affiliation = get_user_input("Affiliation:")

    with open(filename, 'w') as file:
        # Write HTML header with CSS styling
        file.write(f"""
        <html>
        <head>
            <title>CLAIM Checklist Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    color: #333;
                }}
                h1 {{
                    color: #2E4A7D;
                }}
                h2 {{
                    color: #1E2A38;
                    border-bottom: 2px solid #2E4A7D;
                    padding-bottom: 5px;
                }}
                h3 {{
                    color: #1E2A38;
                    margin-top: 20px;
                }}
                p {{
                    margin: 10px 0;
                }}
                .section {{
                    margin-bottom: 20px;
                }}
                .question {{
                    font-weight: bold;
                }}
                .answer {{
                    margin-left: 20px;
                    font-style: italic;
                }}
                .author-info {{
                    margin-top: 20px;
                    padding-top: 10px;
                    border-top: 1px solid #ccc;
                }}
                .citation {{
                    margin-top: 20px;
                    font-size: 0.9em;
                    color: #555;
                }}
            </style>
        </head>
        <body>
            <h1>CLAIM Checklist Report</h1>
            <p class="author-info"><strong>Author:</strong> {author_name}</p>
            <p class="author-info"><strong>Affiliation:</strong> {affiliation}</p>
        """)

        # Populate HTML with content
        current_section = ""
        current_subsection = ""
        for question_number, question, response, section, subsection in responses:
            if section != current_section:
                current_section = section
                file.write(f"<h2>{current_section}</h2>\n")

            if subsection != current_subsection:
                current_subsection = subsection
                if current_subsection:
                    file.write(f"<h3>{current_subsection}</h3>\n")

            file.write(f"<p class='question'>{question_number}. {question}</p>\n")
            file.write(f"<p class='answer'>Answer: {response}</p>\n")

        # Add citation
        citation_text = ("Mongan J, Moy L, Kahn CE Jr. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): "
                         "a guide for authors and reviewers. Radiol Artif Intell 2020; 2(2). https://doi.org/10.1148/ryai.2020200029")
        file.write(f"<p class='citation'><em>{citation_text}</em></p>\n")

        # Write HTML footer
        file.write("""
        </body>
        </html>
        """)

# File name for the HTML report
filename = "CLAIM_Report.html"

# Create the HTML report with user responses
create_html(filename, responses)

print(f"CLAIM Report saved as {filename}")
