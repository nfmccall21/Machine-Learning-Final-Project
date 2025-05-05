# **READ ME**

## Overview

Our project looks at what makes an Instagram caption drive higher engagement, specifically through the lens of caption content. Using a dataset found in "Multimodal Post Attentive Profiling for Influencer Marketing," by Seungbae Kim, Jyun-Yu Jiang, Masaki Nakada, Jinyoung Han and Wei Wang, we iterated over 1.6 million posts from 38,000+ influencers. We combined structured metadata (like hashtags, mentions, and comment counts) with semantic features extracted using DistilBERT embeddings. We trained predictive models to estimate like counts and clustered captions by semantic similarity to uncover content-based egagement patterns. Key findings showed that BERT-derived features outperformed basic metadata, confiming that the tone and semantic content of a caption significantly influence engagement. Captions in clusters assocaited with celebrity/product hype performed best, while niche or aesthetic focused clusters did worse.

## Replication Instructions

1. First you need a dataset of instagram metadata –– gather post images as well if you hope to expand into future direction suggestions.
2. Pre-process the data; this includes clean the caption text of whitespace, and extracting features like like_count, comment_count, mention_count, hashtag_count, caption_length, and aspect_ration.
3. Use HuggingFace's DistilBERT model to embed each caption, as well as the transformers library in Python. Store these 768-dim vectors for each post as additional features.
4. For model training, split the data 80% training / 20% testing. Train a linear regression model using three formats
   a. (baseline) structure features
   b. structured & BERT features (if you have a large dataset, here is where you can play around with truncated versions of the data before performing on the full dataset)
5. Perform KMeans cluserting on BERT embeddings (k=5 is a good starting point). Analyze average likes per cluster to identify high and low performing semantic categories.
6. To evaluate, report RMSE and R^2 scores for each model. We interpreted top regression coefficients and summarized cluster-based performance.

## Future Directions

There were several clear directions for us to extend this work. First we would look to incorporate immage data through CNNs or autoencoders, as we predict this would dramatically increase performance seeing the visual component of Instagram is significant. Another direction would be a recommendation engine that blends caption embeddings with visual aesthetics to suggest caption-image combinations that would perform well. WE would also like to transition away from like count as a metric, hoping to incorporate saves, shares, and comments as a scale of post success and engagement. Finally, profiling influencer types or sponsor cluster could offer more tailored strategies, particularly for different creater niches. 

## Contributions

Throughout this project, we adopted a fully collaborative workflow. We were both equally involved in research, technical elements, and articulating the findings. 
Rather than dividing the workload into strict sections, we found it more beneficial to adopt a programming and co-writing model. This means we chose to work in parallel, most times in the same space or on the phone. Writing code, troubleshooting issues, going over results, and refining our methodology were done together in real-time. 
We alternated drafting and editing responsibilities throughout the writing process, making sure both voices and ideas were integrated and that we both fully understood 100% of the project. Natailie took the lead on portions of the code implementation, while Sarah led the design and development of the poster.
The final submission reflects a joint effort, with no singular ownership over any part. This approach worked best for our work quality and understanding of the methods and findings.
