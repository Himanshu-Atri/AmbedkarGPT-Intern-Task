# Result Analysis

## Evaluation Summary

The average value over the 25-test dataset:

- **Average Recall:** 0.84  
- **Average Precision:** 0.52  
- **Average MRR:** 0.76  

- **Average Rouge F1 Score:** 0.326  
- **Average Cosine Score:** 0.581  
- **Average Bleu Score:** 0.081  


## Retrieval Analysis

A high recall value tells us that most of the time the model was able to find relevant chunks.  
The moderate precision value tells us that the fetched chunks weren’t always relevant.

The cause of this moderate value is the fixed retrieval of 2 chunks. In some test cases, there is only 1 relevant chunk, while in others there are more than 2. There are also a few cases where no relevant chunk exists at all, but Chroma will always return 2 chunks regardless of whether there are more than 2 relevant chunks or fewer than 2 relevant chunks.

A relatively high MRR score confirms that the relevant documents were fetched sooner than the irrelevant ones.

Overall, the script was able to retrieve the most relevant chunks that were available.


### Improvements for Retrieval

- Deciding a threshold in the similarity search to return all chunks that score higher than the given threshold. This will dynamically determine the number of chunks, ensure relevance, and avoid returning any chunk at all when no relevant information exists.


## Generation Analysis (My Findings)

The Rouge F1 score is low because there was not enough overlap between the ground truth and the generated output.  
This happened because even though the answers generated were very similar, they were much longer in length compared to the ground truth. The moderate cosine similarity supports this: the generated outputs captured the ground truth but also included additional information not present in the ground truth.

The BLEU score is very low because it looks for exact phrasing, which isn't very relevant in a QA chatbot. This low value indicates that the phrasing did not match the ground truth.

Overall, the model captures the ground truth and some additional relevant information as well.


### Improvements for Generation

1. Implementing a more strict prompt template to force the model to generate shorter answers similar in length and nature to the test dataset’s ground truth.  
2. Including a few examples in the prompt template to help the model generate output similar in structure and length to the test dataset.  
3. Fine-tuning the model is also an option, but it doesn’t seem necessary and would likely be overkill in this scenario.
