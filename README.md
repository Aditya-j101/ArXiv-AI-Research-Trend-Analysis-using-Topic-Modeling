# ArXiv-AI-Research-Trend-Analysis-using-Topic-Modeling
This project performs an end-to-end analysis of research trends in Artificial Intelligence by applying unsupervised topic modeling to the ArXiv dataset. The primary goal is to identify, interpret, and visualize the major subfields and thematic shifts within AI research published between 2010 and 2025.
Link to Dataset - https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset
Dataset Description: ArXiv Scientific Research Papers Dataset
This dataset is a curated collection of research papers from arXiv, covering various scientific fields such as Artificial Intelligence, Machine Learning, computer science, mathematics and more. It includes titles, abstracts, categories, authors, publications and Updated dates making it useful for various machine learning and NLP tasks. It can be used for several use cases given below

Research Paper Classification – Train a model to predict a paper’s category based on its title or abstract.
Recommendation Systems – Suggest similar papers based on textual content or categories.
Trend Analysis – Analyze how research topics evolve.
Text Summarization – Develop models to generate concise summaries of scientific papers.
Topic Modeling – Identify hidden themes and emerging research areas.
Author Impact Analysis – Track contributions and collaborations across different domains.

##Features
Features
1. End-to-End NLP Pipeline: From raw text to cleaned, lemmatized data ready for modeling.
2. Unsupervised Topic Modeling: Implements Latent Dirichlet Allocation (LDA) to discover hidden thematic structures in the text corpus.
3. Interactive Visualization: Uses pyLDAvis to explore the discovered topics and their associated keywords for effective manual interpretation.
4. Time-Series Trend Analysis: Assigns a dominant topic to each paper and plots the proportional representation of each topic year-over-year.

## Topics Discovered
1. The analysis successfully uncovered distinct and coherent research themes within the AI literature. The key topics identified include:
2. AI Reasoning & Search: Focuses on constraint satisfaction, search algorithms, and probabilistic reasoning like Bayesian networks.
3. Graph Neural Networks & Embeddings: Covers deep learning on graph-structured data and knowledge graph representations.
4. AI Ethics & Healthcare: Relates to the responsible and ethical application of AI in high-stakes domains like medicine.
5. Explainable AI for LLMs (XAI): Centers on making large language models more transparent and interpretable.
6. Deep Reinforcement Learning for Games: Involves training agents to master complex games using reinforcement learning and MCTS.
7. Knowledge Representation & Reasoning (KRR): The classic, logic-based approach to representing knowledge and performing automated reasoning.
8. Decision Making Under Uncertainty: Explores methods like Fuzzy Logic and Rough Set Theory for making optimal choices with incomplete information.
9. Time Series Analysis & Data Mining: Focuses on forecasting, anomaly detection, and pattern recognition in temporal data streams.
10. Automated Planning & Robotics: Covers goal-oriented action planning for robots, often dealing with uncertainty (POMDPs).
11. Autonomous Driving & Traffic Optimization: Relates to the control, simulation, and optimization of intelligent transportation systems.

Of course. Here is a comprehensive README file for your project. You can copy and paste this text into a new file named README.md in your GitHub repository.

ArXiv AI Research Trend Analysis using Topic Modeling
This project performs an end-to-end analysis of research trends in Artificial Intelligence by applying unsupervised topic modeling to the ArXiv dataset. The primary goal is to identify, interpret, and visualize the major subfields and thematic shifts within AI research published between 2010 and 2025.

The final output is a time-series visualization that tracks the prevalence of these topics over time, illustrating the evolution of the AI research landscape.

## Features
End-to-End NLP Pipeline: From raw text to cleaned, lemmatized data ready for modeling.

Unsupervised Topic Modeling: Implements Latent Dirichlet Allocation (LDA) to discover hidden thematic structures in the text corpus.

Interactive Visualization: Uses pyLDAvis to explore the discovered topics and their associated keywords for effective manual interpretation.

Time-Series Trend Analysis: Assigns a dominant topic to each paper and plots the proportional representation of each topic year-over-year.

## Topics Discovered
The analysis successfully uncovered distinct and coherent research themes within the AI literature. The key topics identified include:

AI Reasoning & Search: Focuses on constraint satisfaction, search algorithms, and probabilistic reasoning like Bayesian networks.

Graph Neural Networks & Embeddings: Covers deep learning on graph-structured data and knowledge graph representations.

AI Ethics & Healthcare: Relates to the responsible and ethical application of AI in high-stakes domains like medicine.

Explainable AI for LLMs (XAI): Centers on making large language models more transparent and interpretable.

Deep Reinforcement Learning for Games: Involves training agents to master complex games using reinforcement learning and MCTS.

Knowledge Representation & Reasoning (KRR): The classic, logic-based approach to representing knowledge and performing automated reasoning.

Decision Making Under Uncertainty: Explores methods like Fuzzy Logic and Rough Set Theory for making optimal choices with incomplete information.

Time Series Analysis & Data Mining: Focuses on forecasting, anomaly detection, and pattern recognition in temporal data streams.

Automated Planning & Robotics: Covers goal-oriented action planning for robots, often dealing with uncertainty (POMDPs).

Autonomous Driving & Traffic Optimization: Relates to the control, simulation, and optimization of intelligent transportation systems.

## Tech Stack & Libraries
1. Python 3.10+
2. Data Manipulation & Analysis: Pandas, NumPy
3. NLP: NLTK
4. Machine Learning & Modeling: Scikit-learn
5. Data Visualization: Matplotlib, Seaborn, pyLDAvis
6. Environment: Jupyter Notebook / Google Colab
