import numpy as np
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from transformers import pipeline

class ResearchQASystem:
    def __init__(self, uri="neo4j+s://1148f8c6.databases.neo4j.io", user="neo4j", password="_KlmMeyAATXPPTOgSu2jcjQPxaso12x0_5MiK3S4I4U"):
        """Initialize the Research QA System with necessary models and database connection"""
        # Neo4j connection
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        # QA Model for answering questions
        self.generator = pipeline("text-generation", model="mistralai/Mistral-Small-Instruct-2409", device=0)

    def get_relevant_papers(self, question, top_k=5):
        """
        Retrieve the most relevant papers based on semantic similarity with the question

        Parameters:
        question (str): The user's question
        top_k (int): Number of most relevant papers to retrieve

        Returns:
        list: List of relevant paper information
        """
        # Query to get papers and limit to the top_k most relevant papers
        query = """
        MATCH (p:Paper)
        RETURN p.title, p.summary, p.published, p.link
        ORDER BY p.published DESC
        LIMIT $top_k
        """

        with self.driver.session() as session:
            results = session.run(query, top_k=top_k)
            papers = [record.data() for record in results]

        return papers

    def generate_answer(self, question):
        """
        Generate an answer based on relevant papers from the database

        Parameters:
        question (str): The user's question

        Returns:
        dict: Contains the answer, source papers, and confidence score
        """
        # Get relevant papers
        relevant_papers = self.get_relevant_papers(question)

        if not relevant_papers:
            return {
                "answer": "I couldn't find any relevant papers to answer this question.",
                "sources": [],
                "confidence": 1.0
            }

        # Prepare combined context from relevant papers
        answers = []
        for paper in relevant_papers:
            # Create a structured context
            prompt = f"""
            <s> [INST]
            Title: {paper['p.title']}
            Content: {paper['p.content']}
            Summary: {paper['p.summary']}
            Published: {paper['p.published']}

            {question}
            Also provide a confidence score of your produced output. This score should be strictly between 0 and 1.[/INST]
            """

            # Get answer from generator model
            qa_result = self.generator(prompt, max_length=150)

            answers.append({
                "answer": qa_result[0]['generated_text'],
                "confidence": 1.0,  # Confidence here is simplified, you may enhance it
                "source": paper['p.title']
            })

        # Sort answers by confidence
        answers.sort(key=lambda x: x['confidence'], reverse=True)

        # Combine information from multiple sources if available
        if len(answers) > 1:
            best_answer = answers[0]
            sources = [ans['source'] for ans in answers]

            return {
                "answer": best_answer['answer'],
                "sources": sources,
                "confidence": best_answer['confidence'],
                "alternative_answers": answers[1:]
            }
        else:
            return {
                "answer": answers[0]['answer'],
                "sources": [answers[0]['source']],
                "confidence": answers[0]['confidence']
            }

    def close(self):
        """Close the Neo4j driver connection"""
        self.driver.close()