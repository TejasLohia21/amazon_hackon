import os, math
import torch, torch.nn as nn, torch.utils.data as data
import lightning as L

import clip
from dataset import ImageTextDataset

# DEFINE THE FINETUNING ROUTINE
class ClipFinetuner(L.LightningModule):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, image, text):
        image_features = self.clip_model.encode_image(image)
        text_features = self.clip_model.encode_text(text)

        return image_features, text_features

    def training_step(self, batch, batch_idx):
        images, tokenized_text = batch # images:(batch, channels, width, height), tokenized_text:(batch, tokenizer_dim)

        # get embeddings
        image_features, text_features = self(images, tokenized_text.squeeze(1))

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # PUSH X,Y together and push other vectors away. cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # create targets for binary cross-entropy + binary cross entropy
        targets = torch.eye(logits_per_image.size(0), dtype=torch.float32, device=logits_per_image.device)
        loss = nn.functional.binary_cross_entropy_with_logits(logits_per_image, targets)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':

    # LOAD OPEN AI MODEL
    clip_model, preprocess = clip.load("ViT-B/32")
    clip_model.float()

    # LOAD THE DATA
    data_path = os.getcwd() + '/example_dataset'
    custom_dataset = ImageTextDataset(
        image_folder=data_path, 
        annotation_file=f'{data_path}/annotations.txt', 
        tokenize=clip.tokenize, 
        transform=preprocess
    )

    # SETUP THE FINETUNING
    train_data = data.DataLoader(custom_dataset, batch_size=2, shuffle=True, num_workers=3)
    clip_finetuner = ClipFinetuner(clip_model)

    # PTL TRAINER auto-scales across CPUs, GPUs, etc...
    trainer = L.Trainer(max_steps=1000, log_every_n_steps=2)
    trainer.fit(clip_finetuner, train_data)
    
