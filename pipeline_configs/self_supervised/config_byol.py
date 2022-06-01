from self_supervised import LinearClassifierMethodParams
from self_supervised.model_params import BYOLParams

model_params = BYOLParams(
    dataset_name='aisi',
    encoder_arch='resnet101',
    embedding_dim=2048,
    dim=2048,
    mlp_hidden_dim=4096,
    mlp_normalization='bn',
    lr=0.03,
    batch_size=32,
    lars_eta=0.02,
    lars_warmup_epochs=10,
    max_epochs=400,
    loss_constant_factor=2,
    pretrained=True
)

classifier_params = LinearClassifierMethodParams(
    encoder_arch=model_params.encoder_arch,
    embedding_dim=model_params.embedding_dim,
    dataset_name=model_params.dataset_name,
    batch_size=32,
    lr=0.1,
    weight_decay=1.e-4,
    drop_last_batch=False,
    max_epochs=100,
    pretrained=False
)
