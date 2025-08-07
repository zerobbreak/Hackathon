from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_mongo_client():
    mongo_uri = "mongodb+srv://utshuma6:zpyBvSV2LbdMbvhU@cluster0.fyrnhd0.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        logger.info("✅ MongoDB connection successful.")

        # Use a database (will be created if it doesn't exist)
        db = client["crisis_connect"]

        # Initialize model collections (will be created upon first insert if not present)
        required_collections = ['weatherdata', 'historicaldata', 'alerts']

        for collection_name in required_collections:
            if collection_name not in db.list_collection_names():
                db.create_collection(collection_name)
                logger.info(f"📁 Created collection: {collection_name}")
            else:
                logger.info(f"📁 Collection exists: {collection_name}")

        return client, db

    except ConnectionFailure as e:
        logger.error("❌ MongoDB connection failed: %s", e)
        raise
