from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip,RandomResizedCrop, Resize, ToTensor
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np



## load dataset
train_dataset, test_dataset = load_dataset('fashion_mnist', split=['train[:5000]', 'test[:2000]'])
## split training into training and validation
splits = train_dataset.train_test_split(test_size=0.1)
train_dataset = splits['train']
val_dataset = splits['test']

## checking data
# print(train_dataset.features)
# print(train_dataset[10]['image'])
# print(train_dataset.features['label'])

## encode idx and labels
id2label = {idx:label for idx, label in enumerate(train_dataset.features['label'].names)}
label2id = {label:idx for idx, label in enumerate(train_dataset.features['label'].names)}



##Data preprocessing
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
image_mean = processor.image_mean
image_std = processor.image_std
size = processor.size["height"]
# print(image_mean)
# print(image_std)
# print(size)
normalize = Normalize(mean=image_mean, std=image_std)
## create transform functions
_train_transforms = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize])
_test_transforms = Compose([Resize(size),CenterCrop(size),ToTensor(),normalize])

# image_pixel_values = _train_transforms(train_dataset[10]['img'])  # returns tensor
# print(image_pixel_values)

def train_transforms(data):
    data['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in data['image']]
    return data

def test_transforms(data):
    data['pixel_values'] = [_test_transforms(image.convert("RGB")) for image in data['image']]
    return data

## set tranforms
train_dataset.set_transform(train_transforms)
val_dataset.set_transform(test_transforms)
test_dataset.set_transform(test_transforms)

# #print(train_dataset[0])
# print(train_dataset[:3])

# define custom collate function

def collate_fn(dataset):
  pixel_values = torch.stack([data["pixel_values"] for data in dataset])
  labels = torch.tensor([data['label'] for data in dataset])
  return {'pixel_values': pixel_values, 'labels': labels}

# init data loaders
train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=4)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=4)
test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=4)

# batch = next(iter(train_dataloader))
# print(batch['pixel_values'].shape)
# print(batch['labels'].shape)


## Load ViT model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', id2label=id2label, label2id=label2id)


## set argument for training
args = TrainingArguments(
    "test-Fashion_Mnist",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    logging_dir='logs',
    remove_unused_columns=False,
)

## choose metric
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))


## Setup Trainer and train the model
trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)
trainer.train()

# Evaluate model
outputs = trainer.predict(test_dataset)
# print(outputs.metrics)

## print confusion matrix
y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

labels = train_dataset.features['label'].names
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)
plt.savefig('confusion_matrix.png')
