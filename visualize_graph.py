# visualize_graph.py
"""
Visualize Neo4j knowledge graph using PyVis.
Creates an interactive HTML visualization of the graph structure.
"""
import logging
from neo4j import GraphDatabase
from pyvis.network import Network
import networkx as nx
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

NEO_BATCH = 500  # Number of relationships to fetch / visualize

try:
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    # Test connection
    with driver.session() as session:
        session.run("RETURN 1")
    logger.info("Connected to Neo4j successfully")
except Exception as e:
    logger.error(f"Failed to connect to Neo4j: {e}")
    raise

def fetch_subgraph(tx, limit=500):
    """
    Fetch a subgraph from Neo4j for visualization.

    Args:
        tx: Neo4j transaction object
        limit: Maximum number of relationships to fetch

    Returns:
        List of relationship records
    """
    try:
        q = (
            "MATCH (a:Entity)-[r]->(b:Entity) "
            "RETURN a.id AS a_id, labels(a) AS a_labels, a.name AS a_name, "
            "b.id AS b_id, labels(b) AS b_labels, b.name AS b_name, type(r) AS rel "
            "LIMIT $limit"
        )
        result = list(tx.run(q, limit=limit))
        logger.info(f"Fetched {len(result)} relationships for visualization")
        return result
    except Exception as e:
        logger.error(f"Error fetching subgraph: {e}")
        raise

def build_pyvis(rows, output_html="neo4j_viz.html"):
    """
    Build and save PyVis network visualization.

    Args:
        rows: List of relationship records from Neo4j
        output_html: Output HTML file path
    """
    try:
        if not rows:
            logger.warning("No data to visualize")
            return

        logger.info(f"Building visualization with {len(rows)} relationships")

        # Create network with enhanced options
        net = Network(
            height="900px",
            width="100%",
            notebook=False,
            directed=True,
            bgcolor="#ffffff",
            font_color="#000000"
        )

        # Add physics options for better layout
        net.set_options("""
        {
            "physics": {
                "enabled": true,
                "barnesHut": {
                    "gravitationalConstant": -30000,
                    "centralGravity": 0.3,
                    "springLength": 95,
                    "springConstant": 0.04
                },
                "maxVelocity": 50,
                "minVelocity": 0.1,
                "solver": "barnesHut",
                "timestep": 0.35,
                "stabilization": {"iterations": 150}
            }
        }
        """)

        # Track unique nodes
        nodes_added = set()

        for rec in rows:
            a_id = rec["a_id"]
            a_name = rec["a_name"] or a_id
            b_id = rec["b_id"]
            b_name = rec["b_name"] or b_id
            a_labels = rec["a_labels"]
            b_labels = rec["b_labels"]
            rel = rec["rel"]

            # Add nodes with labels
            if a_id not in nodes_added:
                net.add_node(
                    a_id,
                    label=f"{a_name}\n({','.join(a_labels)})",
                    title=f"{a_name}",
                    color="#97c2fc"
                )
                nodes_added.add(a_id)

            if b_id not in nodes_added:
                net.add_node(
                    b_id,
                    label=f"{b_name}\n({','.join(b_labels)})",
                    title=f"{b_name}",
                    color="#97c2fc"
                )
                nodes_added.add(b_id)

            # Add edge with relationship type
            net.add_edge(a_id, b_id, title=rel, label=rel)

        logger.info(f"Added {len(nodes_added)} unique nodes")

        # Save visualization
        net.show(output_html, notebook=False)
        logger.info(f"Saved visualization to {output_html}")
        print(f"\n✓ Visualization saved to: {output_html}")
        print(f"  Nodes: {len(nodes_added)}")
        print(f"  Edges: {len(rows)}")

    except Exception as e:
        logger.error(f"Error building visualization: {e}")
        raise

def main():
    """
    Main function to generate graph visualization.
    """
    try:
        logger.info(f"Fetching up to {NEO_BATCH} relationships from Neo4j...")

        with driver.session() as session:
            rows = session.execute_read(fetch_subgraph, limit=NEO_BATCH)

        if not rows:
            print("\n⚠️  No data found in Neo4j. Please run load_to_neo4j.py first.")
            return

        build_pyvis(rows)

    except Exception as e:
        logger.error(f"Error in visualization: {e}")
        raise
    finally:
        driver.close()
        logger.info("Neo4j driver closed")


if __name__ == "__main__":
    main()
