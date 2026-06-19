import kuzu
db = kuzu.Database(':memory:')
conn = kuzu.Connection(db)
conn.execute('CREATE NODE TABLE Entity (node_id STRING PRIMARY KEY)')
conn.execute('CREATE REL TABLE RELATES (FROM Entity TO Entity)')
conn.execute('CREATE (n:Entity {node_id: "1"})-[:RELATES]->(m:Entity {node_id: "2"})')
conn.execute('MATCH (n:Entity {node_id: "1"}) DETACH DELETE n')
res = conn.execute('MATCH (n:Entity) RETURN n.node_id')
print("Nodes left:", [r for r in res])
