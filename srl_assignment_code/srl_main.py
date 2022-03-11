import tempfile
from typing import Dict, Iterable, List, Tuple

import allennlp
import torch
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader

from allennlp.models import Model
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import StackedAlternatingLstmSeq2SeqEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util
from allennlp.training.trainer import Trainer
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer, HuggingfaceAdamWOptimizer
from allennlp.training.learning_rate_schedulers.slanted_triangular import SlantedTriangular
from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor

from tools.srl_reader import SrlReader
from tools.srl_model_bert import SrlBert
from tools.srl_model_lstm import SemanticRoleLabelerLSTM
from tools.srl_predictor import SemanticRoleLabelerPredictor


def read_data(reader: DatasetReader, data_paths: Dict) -> Tuple[List[Instance], List[Instance]]:
    print("Reading srl_data")
    training_data = list(reader.read(data_paths['train']))
    validation_data = list(reader.read(data_paths['validation']))
    return training_data, validation_data


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)


def build_dataset_reader(bert_model_name=None) -> DatasetReader:
    if not bert_model_name: 
        return SrlReader()
    else:
        return SrlReader(bert_model_name=bert_model_name)


def build_model(vocab: Vocabulary, type: str = 'lstm') -> Model:
    print("Building the model")
    if type.lower() == 'lstm':
        vocab_size = vocab.get_vocab_size("tokens")
        embedder = BasicTextFieldEmbedder({"tokens": Embedding(embedding_dim=100, num_embeddings=vocab_size)})
        encoder = StackedAlternatingLstmSeq2SeqEncoder(input_size=120, hidden_size=300, num_layers=8)
        return SemanticRoleLabelerLSTM(vocab, embedder, encoder, binary_feature_dim=20, ignore_span_metric=True)
    elif type.lower() == 'bert':
        return SrlBert(vocab, 'bert-base-cased')
    else:
        raise NotImplementedError


def run_training_loop(model_type, train_data, dev_data, serialization_dir):
    vocab = build_vocab(train_data + dev_data)
    vocab.save_to_files(serialization_dir)
    
    model = build_model(vocab, model_type)
    if GPU_IX >= 0: model.to(GPU_IX)

    train_loader, dev_loader = build_data_loaders(train_data, dev_data)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    if model_type.lower() == 'lstm':
        trainer = build_lstm_trainer(model, serialization_dir, train_loader, dev_loader)
    elif model_type.lower() == 'bert':
        trainer = build_bert_trainer(model, serialization_dir, train_loader, dev_loader)
    else:
        raise NotImplementedError
    print("Starting training")
    trainer.train()
    print("Finished training")

    return model


def build_data_loaders(train_data: List[Instance], dev_data: List[Instance]) -> Tuple[DataLoader, DataLoader]:
    train_loader = SimpleDataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = SimpleDataLoader(dev_data, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, dev_loader


def build_bert_trainer(model: Model, serialization_dir: str, train_loader: DataLoader, dev_loader: DataLoader) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = HuggingfaceAdamWOptimizer(model_parameters=parameters, 
                                        parameter_groups=[[["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],], 
                                        lr=5e-5, weight_decay=0.01)  # type: ignore
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=EPOCHS,
        optimizer=optimizer,
        learning_rate_scheduler=SlantedTriangular(optimizer, EPOCHS),
        cuda_device=GPU_IX,
    )
    return trainer


def build_lstm_trainer(model: Model, serialization_dir: str, train_loader: DataLoader, dev_loader: DataLoader) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters, lr=0.001)  # type: ignore
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=EPOCHS,
        optimizer=optimizer,
        cuda_device=GPU_IX,
    )
    return trainer


if __name__ == '__main__':
    # Get Proper Paths
    MODEL_TYPE = 'lstm' # ACCEPTED: 'lstm' | 'bert'
    BERT_MODEL_NAME = 'bert-base-cased' if MODEL_TYPE.lower() == 'bert' else ''
    EPOCHS=1
    BATCH_SIZE=32
    GPU_IX=-1
    TRAIN_PATH = "data/srl_univprop_en.train.jsonl"
    DEV_PATH = "data/srl_univprop_en.dev.jsonl"
    SAVE_MODEL_PATH = f"data/model_srl_{MODEL_TYPE}"

    # Read the Data
    dataset_reader = build_dataset_reader(BERT_MODEL_NAME)
    train_data, dev_data = read_data(dataset_reader, data_paths={'train': TRAIN_PATH, 'validation': DEV_PATH})

    # Train the Model
    model = run_training_loop(MODEL_TYPE, train_data, dev_data, SAVE_MODEL_PATH)
    vocab = model.vocab


    # Predict unseen instances using the model that we just trained
    predictor = SemanticRoleLabelerPredictor(model, dataset_reader, language="en_core_web_sm")

    output = predictor.predict("I am running away from here!")
    print(" ".join(output['words']))
    print(output['words'])
    [print(f"\tVERB: {verb_obj['verb']} | ARGS: {verb_obj['tags']}") for verb_obj in output['verbs']]

    print()

    output = predictor.predict("The paint and wheels looked like glass and the interior looked new!")
    print(" ".join(output['words']))
    print(output['words'])
    [print(f"\tVERB: {verb_obj['verb']} | ARGS: {verb_obj['tags']}") for verb_obj in output['verbs']]
    
