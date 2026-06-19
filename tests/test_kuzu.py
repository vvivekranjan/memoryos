import kuzu
db = kuzu.Database(':memory:')
conn = kuzu.Connection(db)
conn.execute('CREATE NODE TABLE Entity (node_id STRING PRIMARY KEY)')
conn.execute('CREATE REL TABLE RELATES (FROM Entity TO Entity)')

queries = [
    'MATCH (start:Entity)-[r:RELATES*1..2]->(end:Entity) RETURN r',
    'MATCH (start:Entity)-[r:RELATES*1..2]->(end:Entity) RETURN length(r)',
    'MATCH (start:Entity)-[r:RELATES*1..2]->(end:Entity) RETURN r, length(r)',
    'MATCH p = (start:Entity)-[*1..2]->(end:Entity) RETURN length(p)',
    'MATCH (start:Entity)-[r*1..2]->(end:Entity) RETURN r',
]

for q in queries:
    try:
        conn.execute(q)
        print("SUCCESS:", q)
    except Exception as e:
        print("ERROR:", q, "->", e)
