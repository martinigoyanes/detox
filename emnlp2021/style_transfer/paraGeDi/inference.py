import argparse
import logging
import time
import torch
logger = logging.getLogger(__name__)

from gedi_adapter import GediAdapter, NEUTRAL, TOXIC

PARAPHRASER_NAME = 'ceshine/t5-paraphrase-paws-msrp-opinosis'
GEDI_NAME = 's-nlp/gpt2-base-gedi-detoxification'
CLF_NAME = 'SkolkovoInstitute/roberta_toxicity_classifier_v1'

# TODO:
# Use fine-tuned paraphraser on toxic parallel data
# Finetune ROBERTA on toxic dataset to use a classifier for the re-rank step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='',
    				   help="Path to the folder with test data"
    				   )
    parser.add_argument("--output_dir", type=str, default='',
    				   help="Path to the folder to output predictions"
    				   )
    args = parser.parse_args()

    # Setup Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    logging.info(f"Loading data from {args.data_dir} and storing inference results in {args.output_dir}")

    gedi_adapter = GediAdapter(
        paraphraser_name=PARAPHRASER_NAME,
        gedi_name=GEDI_NAME,
        clf_name=None,
        gedi_logit_coef=10, 
        target=NEUTRAL, 
        reg_alpha=3e-5, ub=0.01,
        debug=False,
        device=f"cuda:0" if torch.cuda.is_available() else "cpu"
    )
    gedi_adapter._setup(gedi_name=GEDI_NAME)
    # print_gpu_utilization()

    # Load test data
    with open(args.data_dir, 'r') as f:
        toxic_texts = [line.strip() for line in f.readlines()]

    batch_size = 4
    start = time.time()
    for i in range(0, len(toxic_texts), batch_size):
        text_batch = toxic_texts[i:i+batch_size]
        logger.info(f"From {i} to {i+batch_size}")
        
        preds = gedi_adapter.paraphrase_and_rerank(
            texts=text_batch, 
            max_length='auto', 
            beams=3, 
            rerank=False, 
        )
        logger.info(f"Predictions:\n{preds}")
        # print_gpu_utilization()

        end = time.time()
        logger.info(f"Running time: {end-start}s")

        with open(f"{args.output_dir}/preds.txt", 'a') as f:
            f.write('\n'+'\n'.join(preds))


if __name__ == "__main__":
    main()