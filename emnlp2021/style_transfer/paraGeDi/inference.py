import argparse
import logging
import time
logger = logging.getLogger(__name__)

from gedi_adapter import GediAdapter, NEUTRAL, TOXIC
import text_processing

NEUTRAL = 0
TOXIC = 1
PARAPHRASER_NAME = 'ceshine/t5-paraphrase-paws-msrp-opinosis'
GEDI_NAME = 's-nlp/gpt2-base-gedi-detoxification'
CLF_NAME = 'SkolkovoInstitute/roberta_toxicity_classifier_v1'

# TODO:
# Use fine-tuned paraphraser on toxic parallel data
# Finetune ROBERTA on toxic dataset to use a classifier for the re-rank step

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--not_used", type=str, default='',
    # 				   help="Path to the folder with pretrained model"
    # 				   )
    # args = parser.parse_args()

    # Setup Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    gedi_adapter = GediAdapter(
        paraphraser_name=PARAPHRASER_NAME,
        gedi_name=GEDI_NAME,
        clf_name=None,
        gedi_logit_coef=10, 
        target=NEUTRAL, 
        reg_alpha=3e-5, ub=0.01,
        debug=False,
    )
    gedi_adapter._setup(gedi_name=GEDI_NAME)

    # Load test data
    with open('../../data/test_10k_toxic', 'r') as f:
        test_toxic_data = [line.strip() for line in f.readlines()]

    logger.info(len(test_toxic_data))
    logger.info(test_toxic_data[:5])
    start = time.time()
    preds = gedi_adapter.paraphrase_and_rerank(
        texts=test_toxic_data[:20], 
        max_length='auto', 
        beams=10, 
        rerank=False, 
    )
    end = time.time()
    logger.info(f"Time: {end-start}s")
    logger.info(preds)

if __name__ == "__main__":
    main()