import unittest
import tempfile
import shutil
import os
from pathlib import Path

from utils.note_processor import NoteProcessor
from utils.retrievers import CornellNoteRetriever, ZettelNoteRetriever
from utils.cornell_zettel_memory_system import CornellMethodNote, ZettelNote

import logging

# Set up logger for this module with a more descriptive name
logger = logging.getLogger(__name__)

class TestNoteProcessorIntegration(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for test indices
        self.temp_dir = tempfile.mkdtemp()
        self.cornell_index_path = os.path.join(self.temp_dir, "cornell_index")
        self.zettel_index_path = os.path.join(self.temp_dir, "zettel_index")
        
        # Create real retrievers with temporary paths
        self.cornell_retriever = CornellNoteRetriever(index_path=self.cornell_index_path)
        self.zettel_retriever = ZettelNoteRetriever(index_path=self.zettel_index_path)
        
        # Create the note processor with real retrievers
        self.note_processor = NoteProcessor(
            cornell_retriever=self.cornell_retriever,
            zettel_retriever=self.zettel_retriever
        )
        
        # Sample text for testing
        self.sample_text = """
        # Machine Learning Fundamentals
Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. This field has gained significant attention in recent years due to its ability to analyze and interpret complex data, making it a crucial tool for various industries such as healthcare, finance, and technology. Machine learning enables computers to automatically improve their performance on a task without being explicitly programmed, making it a key driver of innovation and growth.

## Supervised Learning
Supervised learning is a type of machine learning that uses labeled data to train models. In this approach, the algorithm is provided with a dataset that contains input data and corresponding output labels. The goal of supervised learning is to learn a mapping between input data and output labels, so the model can make predictions on new, unseen data. Common algorithms used in supervised learning include:

* **Linear Regression**: A linear regression model predicts a continuous output variable based on one or more input features. It is commonly used for forecasting and regression tasks.
* **Decision Trees**: A decision tree is a tree-like model that splits data into subsets based on feature values. It is often used for classification and regression tasks.
* **Support Vector Machines (SVMs)**: An SVM is a powerful algorithm that can be used for classification and regression tasks. It works by finding the hyperplane that maximally separates the classes in the feature space.

Supervised learning has numerous applications, including:

* Image classification
* Sentiment analysis
* Speech recognition
* Predictive maintenance

## Unsupervised Learning
Unsupervised learning is a type of machine learning that works with unlabeled data to find patterns. In this approach, the algorithm is provided with a dataset that contains only input data, and the goal is to identify underlying structures or relationships in the data. Common algorithms used in unsupervised learning include:

* **K-means clustering**: K-means clustering is a technique that groups similar data points into clusters based on their features. It is commonly used for customer segmentation and gene expression analysis.
* **Principal Component Analysis (PCA)**: PCA is a dimensionality reduction technique that transforms high-dimensional data into lower-dimensional data while retaining most of the information. It is often used for data visualization and feature extraction.
* **Hierarchical clustering**: Hierarchical clustering is a technique that builds a hierarchy of clusters by merging or splitting existing clusters. It is commonly used for gene expression analysis and customer segmentation.

Unsupervised learning has numerous applications, including:

* Customer segmentation
* Anomaly detection
* Gene expression analysis
* Recommendation systems

## Semi-Supervised Learning
Semi-supervised learning is a type of machine learning that combines labeled and unlabeled data to train models. This approach is useful when labeled data is scarce or expensive to obtain. Semi-supervised learning algorithms can be used for classification, regression, and clustering tasks.

## Reinforcement Learning
Reinforcement learning is a type of machine learning that involves an agent learning to take actions in an environment to maximize a reward. The agent learns through trial and error, and the goal is to develop a policy that maps states to actions. Reinforcement learning has numerous applications, including:

* Robotics
* Game playing
* Autonomous vehicles
* Recommendation systems

In conclusion, machine learning is a powerful tool that has numerous applications in various industries. Understanding the fundamentals of machine learning, including supervised, unsupervised, semi-supervised, and reinforcement learning, is crucial for developing effective machine learning models that can drive innovation and growth.
        """
    
    def tearDown(self):
        # Clean up temporary directories
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_processing(self):
        """Test the entire note processing workflow with real data"""
        # Process the sample text
        cornell_note, zettel_notes = self.note_processor.process_text(
            text=self.sample_text,
            source_id="ml-fundamentals",
            source_title="Machine Learning Fundamentals"
        )
        logger.info(f"Created Cornell note: {cornell_note.summary}")
        logger.info(f"Created number of Zettel notes: {len(zettel_notes)}")
        logger.info(f"Zettel notes during integration testing: {[note.title for note in zettel_notes]}")
        
        # Verify Cornell note was created correctly
        self.assertIsInstance(cornell_note, CornellMethodNote)
        self.assertEqual(cornell_note.source_id, "ml-fundamentals")
        self.assertEqual(cornell_note.source_title, "Machine Learning Fundamentals")
        self.assertIsInstance(cornell_note.title, str)
        
        # Verify Zettel notes were created
        self.assertGreater(len(zettel_notes), 0)
        for note in zettel_notes:
            self.assertIsInstance(note, ZettelNote)
        
        # Verify notes can be retrieved by source
        retrieved_notes = self.note_processor.get_notes_by_source("ml-fundamentals")
        self.assertEqual(len(retrieved_notes), 1)
        retrieved_cornell, retrieved_zettels = retrieved_notes[0]
        
        # Verify retrieved Cornell note matches the original
        self.assertEqual(retrieved_cornell.id, cornell_note.id)
        self.assertEqual(retrieved_cornell.title, cornell_note.title)
        
        # Verify all Zettel notes were retrieved
        self.assertEqual(len(retrieved_zettels), len(zettel_notes))
        
        # Test search functionality
        search_results = self.note_processor.zettel_retriever.search("supervised learning", k=5)
        self.assertGreater(len(search_results), 0)
    
    def test_batch_processing(self):
        """Test batch processing of multiple texts"""
        # Create multiple sample texts
        texts = [
            self.sample_text,
            """# Deep Learning Overview
Deep learning is a subset of machine learning that uses neural networks with multiple layers. These neural networks are designed to mimic the human brain's ability to learn and interpret data, making them particularly effective in tasks such as image recognition, speech recognition, and natural language processing. The key characteristic of deep learning is the use of multiple layers, which allows the network to learn complex patterns and representations of the input data.

## Convolutional Neural Networks
CNNs are specialized for processing grid-like data such as images. They are designed to take advantage of the spatial hierarchies of images, using convolutional and pooling layers to extract features and reduce the dimensionality of the data. CNNs have been widely used in applications such as:
* Image classification: CNNs can be trained to recognize objects, scenes, and activities in images.
* Object detection: CNNs can be used to detect and locate objects within images.
* Image segmentation: CNNs can be used to segment images into different regions or objects.
* Image generation: CNNs can be used to generate new images, such as in image-to-image translation tasks.

The architecture of a CNN typically consists of several convolutional and pooling layers, followed by fully connected layers. The convolutional layers apply filters to the input data, scanning the data in a sliding window fashion to extract features. The pooling layers reduce the spatial dimensions of the data, helping to reduce the number of parameters and the risk of overfitting.

## Recurrent Neural Networks
RNNs are designed for sequential data processing, such as time series data, speech, or text. They are particularly effective in tasks that require the network to maintain a hidden state over time, such as:
* Language modeling: RNNs can be trained to predict the next word in a sequence of text.
* Speech recognition: RNNs can be used to recognize spoken words and phrases.
* Time series forecasting: RNNs can be used to predict future values in a time series.
* Machine translation: RNNs can be used to translate text from one language to another.

The architecture of an RNN typically consists of a recurrent layer, where the output from the previous time step is fed back into the network as input. This allows the network to maintain a hidden state over time, capturing long-term dependencies in the data. However, RNNs can suffer from vanishing gradients, where the gradients used to update the network's weights become smaller as they are backpropagated through time. This can make it difficult to train RNNs on long sequences of data.

There are several variants of RNNs, including:
* Long Short-Term Memory (LSTM) networks: LSTMs use memory cells to store information over long periods of time, helping to mitigate the vanishing gradient problem.
* Gated Recurrent Units (GRUs): GRUs use gates to control the flow of information into and out of the network, helping to reduce the computational cost of the network.
* Bidirectional RNNs: Bidirectional RNNs process the input data in both the forward and backward directions, helping to capture more context and improve performance on tasks such as language modeling and machine translation.
"""
        ]
        
        # Process each text
        results = []
        for i, text in enumerate(texts):
            cornell_note, zettel_notes = self.note_processor.process_text(
                text,
                source_id=f"source-{i}",
                source_title=f"Title {i}"
            )
            results.append((cornell_note, zettel_notes))
            logger.info(f"Cornell note content: {cornell_note.summary}")
            logger.info(f"Zettel notes during integration testing: {[note.title for note in zettel_notes]}")
        
        # Verify each result
        for i, (cornell_note, zettel_notes) in enumerate(results):
            self.assertIsInstance(cornell_note, CornellMethodNote)
            self.assertEqual(cornell_note.source_id, f"source-{i}")
            self.assertEqual(cornell_note.source_title, f"Title {i}")
            self.assertGreater(len(zettel_notes), 0)
            for note in zettel_notes:
                self.assertIsInstance(note, ZettelNote)