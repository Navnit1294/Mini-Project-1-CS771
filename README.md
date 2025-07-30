# Feature Fusion: Binary Classification with Multi-Modal Data

## Overview

The goal of this project was to explore binary classification using three distinct datasets derived from the same source but represented with different feature types — emoticons, deep embeddings, and digit sequences. The project was divided into two main tasks: evaluating model performance on each dataset independently and then combining all datasets to examine if feature fusion improves performance.

## Team

- Navnit Patel (220701)
- Lakshyta Mahajan (220581)
- Umang Garg (221156)

## Motivation

Different feature representations often carry complementary information. We aimed to explore whether combining such heterogeneous representations can improve the generalization and robustness of binary classifiers, while also analyzing the performance trade-offs in terms of training data size, model complexity, and accuracy.

## Datasets

1. **Emoticon Dataset**: 13 categorical emoticons per sample.
2. **Deep Features Dataset**: 13×786 numerical matrix (likely from deep network embeddings).
3. **Text Sequence Dataset**: Sequences of 50 digits representing input strings.

Each dataset was divided into training, validation, and test sets. Test labels were hidden and used for final evaluation.

---

## Task I: Independent Dataset Modeling

Each dataset was preprocessed individually and used to train binary classifiers. Key steps included:

### Emoticon Dataset

- **Preprocessing**: Unicode conversion → One-hot encoding.
- **Model**: Multinomial Naive Bayes.
- **Best Configuration**: Trained with 80% data, `alpha = 1`.
- **Outcome**: High validation accuracy with efficient training.

### Deep Features Dataset

- **Preprocessing**: Unicode → Embedding transformation.
- **Model**: Multinomial Naive Bayes.
- **Best Configuration**: Trained with 40% data, `alpha = 10`.
- **Outcome**: Demonstrated effective generalization even with limited data.

### Text Sequence Dataset

- **Preprocessing**: Digit encoding → Padding to length 50.
- **Model**: Recurrent Neural Network (RNN) with embedding + SimpleRNN layers.
- **Best Configuration**: Trained with 100% data.
- **Outcome**: Highest accuracy among all models for sequence-based data.

---

## Task II: Feature Fusion

All three datasets were combined to build a unified feature representation. Two models were evaluated:

### Neural Network

- **Architecture**: Fully connected layers with 128 → 64 → 32 neurons.
- **Training**: Trained on combined input vectors.
- **Performance**: Achieved up to **99% accuracy** with 100% data.
- **Drawback**: High computational cost due to 10,000+ trainable parameters.

### Multinomial Naive Bayes

- **Configuration**: `alpha = 0.1`, trained with 40% data.
- **Performance**: Achieved ~86% accuracy.
- **Advantage**: Low computation time, good generalization.

---

## Conclusion

The project highlighted the potential of using multiple feature modalities to enhance binary classification performance. While deep models provided superior accuracy, simpler models like Naive Bayes offered competitive results with significantly lower computation costs. Feature fusion proved effective, especially when paired with appropriate model architectures and preprocessing techniques.


