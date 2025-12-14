"""Quick test script for improved retrieval."""

from src.query_processor import QueryProcessor

def test_query_processing():
    processor = QueryProcessor()
    
    test_queries = [
        "How is the economy performing this year?",
        "What are the latest technology news? ",
        "Tell me about sports updates",
        "COVID vaccine news"
    ]
    
    print("🧪 Testing Query Processing\n")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n📝 Original Query: {query}")
        processed = processor.process(query)
        print(f"   🎯 Topic: {processed['topic']}")
        print(f"   🔑 Keywords: {', '.join(processed['keywords'])}")
        print(f"   ➕ Expanded: {processed['expanded']}")
        print(f"   🏷️  Categories: {', '.join(processed['category_filters'])}")
        print("-" * 80)

if __name__ == "__main__":
    test_query_processing()