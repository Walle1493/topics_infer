python lda.py \
    --stopwords_path data/stopwords.txt \
    --input_path data/original_corpus_minus.txt \
    --test_path data/test_set.txt \
    --predict_path data/predict_document.txt \
    --topics_path output/top_topics.csv \
    --infer_path output/topics_infer.csv \
    --test_output output/topics_test.csv \
    --predict_output output/topic_predict.csv \
    --num_topics 10 \
    --num_words 6 \
    --dictionary_file pick/dictionary.pkl \
    --lda_file pick/lda.pkl \
    --do_test \
    --do_predict \