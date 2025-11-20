# import neccessay libraries and main.py
import main
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm



# calculating recall for retrieved documents
def recall(retrieved, truth):
  if len(truth) == 0 or len(retrieved) == 0:
        return 0.0
  hits = sum(1 for item in retrieved if item in truth)
  return hits / len(truth)

# calculating precision for retrieved documents
def precision(retrieved, truth):
    if len(truth) == 0 or len(retrieved) == 0:
        return 0.0
    hits = sum(1 for item in retrieved if item in truth)
    return hits / len(retrieved)

# calculating mrr for retrieved documents
def mrr(retrieved, truth):
    for idx, item in enumerate(retrieved):
        if item in truth:
            return 1.0 / (idx + 1)
    return 0.0


rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# calculating f1 score for model genrated answer
def rouge_f1_score(reference, generated):
    return rouge.score(reference, generated)['rougeL'].fmeasure

# calculating cosine similarity score for model genrated answer
def cosine_score(reference, generated):
    emb_reference = main.embeddings.embed_query(reference)
    emb_generated = main.embeddings.embed_query(generated)
    return cosine_similarity([emb_reference], [emb_generated])[0][0]

# calculating BLEU score for model genrated answer
def bleu_score(reference, generated):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference.split()], generated.split(), smoothing_function=smoothie)

# loading test dataset
with open("data/test_dataset.json", "r") as f:
    test_dataset_dict = json.load(f)
    test_dataset = test_dataset_dict["test_questions"]

# for storing test results
test_results = []

# for calculating average metrics later
total_recall = 0
total_precision = 0
total_mrr = 0

total_rouge_f1_score = 0
total_cosine_score = 0
total_bleu_score = 0


# test dataset evaluation loop
for idx, test_item in tqdm(enumerate(test_dataset), total=len(test_dataset)):

  query = test_item['question']
  correct_answer = test_item['ground_truth']
  correct_docs = test_item['source_documents']
  isAnswerable = test_item['answerable']

  output, docs = main.get_response(query)

  #evaluating retrived documents
  current_recall = recall(docs, correct_docs)
  current_precision = precision(docs, correct_docs)
  current_mrr = mrr(docs, correct_docs)

  total_recall += current_recall
  total_precision += current_precision
  total_mrr += current_mrr

  #evaluating model's output
  current_rouge_f1_score = rouge_f1_score(correct_answer, output)
  current_cosine_score = cosine_score(correct_answer, output)
  current_bleu_score = bleu_score(correct_answer, output)

  total_rouge_f1_score += current_rouge_f1_score
  total_cosine_score += current_cosine_score
  total_bleu_score += current_bleu_score

  test_results.append({
      "id" : idx+1,
      "recall" : current_recall,
      "precision" : current_precision,
      "MRR" : current_mrr,
      "ROUGE-L Score" : current_rouge_f1_score,
      "Cosine Similarity" : current_cosine_score,
      "BLEU Score" : current_bleu_score,
      "question" : query,
      "correct answer" : correct_answer,
      "model output" : output

  })

  tqdm.write(f"\n\nTest Set {idx+1}")
  tqdm.write(f"Current Recall: {current_recall}")
  tqdm.write(f"Current Precision: {current_precision}")
  tqdm.write(f"Current MRR: {current_mrr}")
  tqdm.write("\n")
  tqdm.write(f"Current Rouge F1 Score: {current_rouge_f1_score}")
  tqdm.write(f"Current Cosine Score: {current_cosine_score}")
  tqdm.write(f"Current Bleu Score: {current_bleu_score}")

  tqdm.write(f"Question: {query}")
  tqdm.write(f"Correct Answer: {correct_answer}")
  tqdm.write(f"Model Output: {output}")

print("Average Recall:", (total_recall/len(test_dataset)))
print("Average Precision:", (total_precision/len(test_dataset)))
print("Average MRR:", (total_mrr/len(test_dataset)))

print("Average Rouge F1 Score:", (total_rouge_f1_score/len(test_dataset)))
print("Average Cosine Score:", (total_cosine_score/len(test_dataset)))
print("Average Bleu Score:", (total_bleu_score/len(test_dataset)))

# storing test result in json format
with open("test_results.json", "w") as f:
    json.dump(test_results, f, indent=4)