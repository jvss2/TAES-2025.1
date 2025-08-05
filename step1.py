import random
import logging

from config import (
    DATASET_NAME, DATASET_SPLIT, MIN_EXECUTED_LINES, SAMPLE_SIZE,
    LLM_BACKEND, LLM_MODEL, HF_DEVICE
)
from models import ModelManager
from dataset_processor import DatasetProcessor
from evaluator import ModelEvaluator


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(__name__)


def main():
    logger = setup_logging()
    
    try:
        logger.info("Starting code model evaluation")
        
        model = ModelManager(LLM_BACKEND, LLM_MODEL, HF_DEVICE)
        processor = DatasetProcessor()
        evaluator = ModelEvaluator(model)
        
        samples = processor.load_dataset(DATASET_NAME, DATASET_SPLIT, MIN_EXECUTED_LINES)
        
        random.shuffle(samples)
        if SAMPLE_SIZE:
            samples = samples[:SAMPLE_SIZE]
            logger.info(f"Using random sample of {len(samples)} examples")
        
        results = evaluator.evaluate_samples(samples)
        evaluator.print_summary(results)
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()