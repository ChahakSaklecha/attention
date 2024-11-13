import streamlit as st
import torch
import requests
import xml.etree.ElementTree as ET
import torch
import outlines
from ResearchDatabase import ResearchQASystem, Output
from neo4j import GraphDatabase
from paper import Paper_cls
from transformers import  AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from neo4j import GraphDatabase


def main():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = outlines.models.transformers("mistralai/Ministral-8B-Instruct-2410", model_kwargs={'quantization_config': quantization_config}, device="cuda")
    generator = outlines.generate.json(model, Output)

    # Step 1: Ask the user what type of question they have (Summary or Q&A)
    question_type = st.radio("What type of question do you have?", ("Summary", "Q&A"))

    # Step 2: Ask for the topic
    topic = st.text_input(f"Enter a topic to search for {question_type} related papers:")

    if topic:
        cls = Paper_cls(topic)
        papers = cls.get_arxiv_papers()

        if papers:
            st.subheader(f"Papers related to {topic}:")
            paper_titles = [paper['title'] for paper in papers]

            if question_type == "Summary":
                # For Summary, select a paper to display its details
                selected_title = st.selectbox("Select a paper title for summary", paper_titles)
                if selected_title:
                    selected_paper = next(paper for paper in papers if paper['title'] == selected_title)
                    st.write(f"**Title**: {selected_paper['title']}")
                    st.write(f"**Published**: {selected_paper['published']}")
                    st.write(f"**Authors**: {', '.join(selected_paper['authors'])}")
                    st.write(f"**Summary**: {selected_paper['summary']}")
                    st.write(f"[Read more]({selected_paper['link']})")

            elif question_type == "Q&A":
                # Display Q&A interface for a selected paper
                selected_title = st.selectbox("Relevant Papers", paper_titles)
                if selected_title:
                    selected_paper = next(paper for paper in papers if paper['title'] == selected_title)
                    st.write(f"**Title**: {selected_paper['title']}")
                    st.write(f"**Published**: {selected_paper['published']}")
                    st.write(f"**Authors**: {', '.join(selected_paper['authors'])}")
                    st.write(f"**Summary**: {selected_paper['summary']}")
                    st.write(f"[Read more]({selected_paper['link']})")


                    question = st.text_input("Enter your question:")

                    if st.button("Get Answer"):
                        if question:
                            qa_system = ResearchQASystem(generator)

                            st.write(f"**Question**: {question}")
                            # Generate the answer using the Q&A system
                            result = qa_system.generate_answer(question)

                            # Display main answer
                            st.subheader("Answer")
                            st.write(result['answer'])

                            # Display sources
                            st.subheader("Sources")
                            st.write(", ".join(result['sources']))

                            # Display confidence
                            st.subheader("Confidence")
                            st.write(f"{result['confidence']:.2%}")

                            # Display alternative answers if available
                            if 'alternative_answers' in result:
                                st.subheader("Alternative Answers")
                                for alt in result['alternative_answers'][:2]:  # Show top 2 alternatives
                                    st.write(f"- {alt['answer']} (Confidence: {alt['confidence']:.2%})")
                            
        else:
            st.write("No papers found for this topic.")


if __name__ == "__main__":
    main()



