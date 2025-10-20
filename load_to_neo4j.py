# load_to_neo4j.py
"""
Load Vietnam travel dataset into Neo4j knowledge graph.
Creates nodes with Entity labels and establishes relationships.
"""
import json
import logging
from neo4j import GraphDatabase
from tqdm import tqdm
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_FILE = "vietnam_travel_dataset.json"

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

def create_constraints(tx):
    """
    Create uniqueness constraints for Entity nodes.

    Args:
        tx: Neo4j transaction object
    """
    try:
        # Generic uniqueness constraint on id for node label Entity
        tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
        logger.info("Created uniqueness constraint on Entity.id")
    except Exception as e:
        logger.warning(f"Constraint creation warning (may already exist): {e}")

def upsert_node(tx, node):
    """
    Insert or update a node in Neo4j.

    Args:
        tx: Neo4j transaction object
        node: Node dictionary with id, type, and other properties
    """
    try:
        # Use label from node['type'] and always add :Entity label
        node_type = node.get("type", "Unknown")
        labels = [node_type, "Entity"]
        label_cypher = ":" + ":".join(labels)

        # Keep a subset of properties (exclude connections to avoid nested objects)
        props = {k: v for k, v in node.items() if k not in ("connections",)}

        # Upsert node with properties
        tx.run(
            f"MERGE (n{label_cypher} {{id: $id}}) "
            "SET n += $props",
            id=node["id"],
            props=props
        )

    except Exception as e:
        logger.error(f"Error upserting node {node.get('id', 'unknown')}: {e}")
        raise

def create_relationship(tx, source_id, rel):
    """
    Create a relationship between two nodes.

    Args:
        tx: Neo4j transaction object
        source_id: Source node ID
        rel: Relationship dictionary with 'relation' type and 'target' ID
    """
    try:
        rel_type = rel.get("relation", "RELATED_TO")
        target_id = rel.get("target")

        if not target_id:
            return

        # Create relationship if both nodes exist
        cypher = (
            "MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            "RETURN r"
        )

        result = tx.run(cypher, source_id=source_id, target_id=target_id)

        # Check if relationship was created
        if not result.single():
            logger.debug(f"Skipped relationship {source_id} -> {target_id} (nodes may not exist)")

    except Exception as e:
        logger.warning(f"Error creating relationship {source_id} -[{rel.get('relation')}]-> {rel.get('target')}: {e}")

def main():
    """
    Main function to load data into Neo4j.
    """
    try:
        logger.info(f"Loading data from {DATA_FILE}")

        # Load dataset
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            nodes = json.load(f)

        logger.info(f"Loaded {len(nodes)} nodes from dataset")

        with driver.session() as session:
            # Create constraints
            logger.info("Creating constraints...")
            session.execute_write(create_constraints)

            # Upsert all nodes
            logger.info("Creating nodes...")
            successful_nodes = 0
            failed_nodes = 0

            for node in tqdm(nodes, desc="Creating nodes"):
                try:
                    session.execute_write(upsert_node, node)
                    successful_nodes += 1
                except Exception as e:
                    logger.error(f"Failed to create node {node.get('id')}: {e}")
                    failed_nodes += 1

            logger.info(f"Nodes: {successful_nodes} created, {failed_nodes} failed")

            # Create relationships
            logger.info("Creating relationships...")
            successful_rels = 0
            failed_rels = 0

            for node in tqdm(nodes, desc="Creating relationships"):
                conns = node.get("connections", [])
                for rel in conns:
                    try:
                        session.execute_write(create_relationship, node["id"], rel)
                        successful_rels += 1
                    except Exception as e:
                        failed_rels += 1

            logger.info(f"Relationships: {successful_rels} created, {failed_rels} failed")

        # Get final stats
        with driver.session() as session:
            node_count = session.run("MATCH (n:Entity) RETURN count(n) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]

            logger.info(f"Final stats: {node_count} nodes, {rel_count} relationships")

        print("\n✓ Done loading data into Neo4j!")
        print(f"  Nodes: {node_count}")
        print(f"  Relationships: {rel_count}")

    except FileNotFoundError:
        logger.error(f"Data file not found: {DATA_FILE}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in data file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during Neo4j load: {e}")
        raise
    finally:
        driver.close()
        logger.info("Neo4j driver closed")


if __name__ == "__main__":
    main()
