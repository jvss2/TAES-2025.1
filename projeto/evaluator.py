import json
import logging
from typing import List, Dict, Any

from tqdm import tqdm

from data_structures import PredictionResult
from dataset_processor import DatasetProcessor
from models import ModelManager
from config import PLACEHOLDER, OUTPUT_FILE

logger = logging.getLogger(__name__)


class ModelEvaluator:
    
    def __init__(self, model: ModelManager):
        self.model = model
        self.processor = DatasetProcessor()
    
    def evaluate_samples(self, samples: List[Dict[str, Any]]) -> List[PredictionResult]:
        results = []
        
        logger.info(f"Starting evaluation of {len(samples)} samples")
        logger.info(f"Configuration: backend={self.model.backend}, model={self.model.model_name}")
        logger.info("Mode: Next line prediction")
        
        for idx, sample in enumerate(tqdm(samples, desc="Processing samples")):
            try:
                self._log_sample_header(idx + 1, len(samples))
                
                context, target_line = self.processor.prepare_prompt(
                    sample.get("function", ""),
                    sample.get("executed_lines", []),
                    PLACEHOLDER,
                    sample.get("docstring", "")
                )
                
                line_number = len(context.split('\n')) + 1
                predicted_line, pavg, ptot = self.model.generate(context)
                is_correct = predicted_line.strip() == target_line.strip()
                
                self._log_sample_details(context, target_line, predicted_line, is_correct, line_number)
                
                result = PredictionResult(
                    sample_id=idx + 1,
                    context=context,
                    original_line=target_line,
                    predicted_line=predicted_line,
                    pavg=pavg,
                    ptot=ptot,
                    is_correct=is_correct
                )
                results.append(result)

                if len(results) % 1000 == 0:
                    output_path = f"{OUTPUT_FILE}_{len(results)}.json" 
                    self.save_results(results, output_path)

            except Exception as e:
                logger.error(f"Failed to process sample {idx + 1}: {e}")

        return results
    
    def _log_sample_header(self, sample_num: int, total: int):
        separator = "=" * 60
        logger.info(f"\n{separator}")
        logger.info(f"PROCESSING SAMPLE {sample_num}/{total}")
        logger.info(f"{separator}")

    def _log_sample_details(self, context: str, target_line: str, predicted_line: str, is_correct: bool, line_number: int):
        logger.info(f"TARGET LINE FOR PREDICTION (line {line_number}):")
        logger.info(f"   '{target_line}'")
        logger.info(f"")
        logger.info(f"MODEL PREDICTION:")
        logger.info(f"   '{predicted_line}'")
        logger.info(f"")
        logger.info(f"RESULT: {'CORRECT' if is_correct else 'INCORRECT'}")
        logger.info(f"")
        logger.info(f"CONTEXT PROVIDED TO MODEL:")
        for i, line in enumerate(context.split('\n'), 1):
            logger.info(f"   {i:2d}: {line}")
        logger.info(f"   {line_number:2d}: ??? <- NEXT LINE TO BE PREDICTED")

    def save_results(self, results: List[PredictionResult], filepath: str):
        logger.info(f"Saving {len(results)} results to '{filepath}'")
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump([result.to_dict() for result in results], f, indent=2, ensure_ascii=False)
            logger.info("Results saved successfully")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

    def print_summary(self, results: List[PredictionResult]):
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        separator = "=" * 60
        logger.info(f"\n{separator}")
        logger.info(f"RESULTS SUMMARY")
        logger.info(f"{separator}")
        logger.info(f"Total samples processed: {total}")
        logger.info(f"Correct predictions: {correct}")
        logger.info(f"Incorrect predictions: {total - correct}")
        logger.info(f"Accuracy: {accuracy:.1f}%")
        logger.info(f"{separator}")