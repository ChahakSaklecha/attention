import requests
import xml.etree.ElementTree as ET
from neo4j import GraphDatabase

class Paper_cls:
  def __init__(self, topic):
    self.topic = topic
    
  def get_arxiv_papers(self, max_results=10):
      base_url = "http://export.arxiv.org/api/query?"
      query = f"search_query=all:{self.topic}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
      response = requests.get(base_url + query)
      if response.status_code != 200:
          print("Failed to retrieve data")
          return []

      # Parse the response XML to get paper information
      papers = []
      root = ET.fromstring(response.content)
      for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
          paper = {
              'title': entry.find('{http://www.w3.org/2005/Atom}title').text,
              'summary': entry.find('{http://www.w3.org/2005/Atom}summary').text,
              'authors': [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')],
              'published': entry.find('{http://www.w3.org/2005/Atom}published').text,
              'link': entry.find('{http://www.w3.org/2005/Atom}id').text
          }
          papers.append(paper)

      return papers

# Neo4j Connection Class
class ArxivNeo4j:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_papers_by_topic(self, topic):
        with self.driver.session() as session:
            query = """
                MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic)
                WHERE t.name = $topic
                RETURN p.title AS title, p.summary AS summary, p.published AS published
            """
            result = session.run(query, topic=topic)
            papers = []
            for record in result:
                papers.append({
                    'title': record['title'],
                    'summary': record['summary'],
                    'published': record['published']
                })
            return papers
