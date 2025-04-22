from langchain.tools import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Google Search API wrapper
search = GoogleSerperAPIWrapper()

@tool
def web_search(query: str) -> List[Dict[str, Any]]:
    """Perform a web search and return relevant results.
    
    Args:
        query: Search query string
        
    Returns:
        List of search results with titles and URLs
    """
    raw_results = search.run(query)
    if isinstance(raw_results, str):
        return [{"title": "Search result", "link": raw_results}]
    elif isinstance(raw_results, list):
        return [{"title": r.get('title', ''), "link": r.get('link', '')} for r in raw_results]
    else:
        return [{"title": "Search result", "link": str(raw_results)}]

@tool
def web_scrape(url: str) -> str:
    """Scrape content from a given URL.
    
    Args:
        url: URL to scrape content from
        
    Returns:
        Extracted text content from the webpage
    """
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    return "\n\n".join([doc.page_content for doc in splits])

# Create browsing agent
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0)
browsing_agent = create_react_agent(llm, tools=[web_search, web_scrape])

if __name__ == "__main__":
    # Example usage
    print("Web browsing agent created successfully.")
    print("Available tools:")
    print("- web_search(query)")
    print("- web_scrape(url)")
    
    # Test cases
    print("\nTesting web_search:")
    search_results = web_search("latest AI news 2024")
    print(f"Found {len(search_results)} search results")
    for i, result in enumerate(search_results[:3]):
        print(f"{i+1}. {result.get('title')}")
        print(f"   {result.get('link')}")
    
    print("\nTesting web_scrape:")
    test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    scraped_content = web_scrape(test_url)
    print(f"Scraped {len(scraped_content)} characters from {test_url}")
    print(f"First 200 chars: {scraped_content[:200]}...")