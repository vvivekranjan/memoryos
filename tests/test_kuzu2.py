import kuzu
db = kuzu.Database(':memory:')
conn = kuzu.Connection(db)
conn.execute('CREATE NODE TABLE Entity (node_id STRING PRIMARY KEY)')
conn.execute('CREATE REL TABLE RELATES (FROM Entity TO Entity)')

queries = [
    'MATCH (start:Entity)-[r:RELATES*1..2]->(end_node:Entity) RETURN start.node_id, end_node.node_id',
    'MATCH (start:Entity)-[r:RELATES*1..2]->(target:Entity) RETURN start.node_id, target.node_id',
    'MATCH p = (start:Entity)-[r:RELATES*1..2]->(target:Entity) RETURN start.node_id, target.node_id, length(r)',
]

for q in queries:
    try:
        conn.execute(q)
        print("SUCCESS:", q)
    except Exception as e:
        print("ERROR:", q, "->", e)
