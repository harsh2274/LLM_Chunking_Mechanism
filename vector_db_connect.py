from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "neo4j://localhost:7687",  # or your instance URI
    auth=("neo4j", "123456789")
)

def create_chunk(tx, idx, text):
    tx.run(
        "CREATE (c:Chunk {id: $id, text: $text})",
        id=idx,
        text=text
    )

with driver.session() as session:
    for i, chunk in enumerate(chunks):
        session.write_transaction(create_chunk, i, chunk)