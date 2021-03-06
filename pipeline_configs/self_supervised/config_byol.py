from self_supervised import LinearClassifierMethodParams
from self_supervised.model_params import BYOLParams


model_params = BYOLParams(
    dataset_name='aisi',
    encoder_arch='inception_resnet_v2',
    embedding_dim=1536,
    dim=2048,
    mlp_hidden_dim=2048,
    mlp_normalization='bn',
    lr=0.00315,
    batch_size=32,
    lars_eta=0.0846,
    lars_warmup_epochs=10,
    max_epochs=300,
    loss_constant_factor=2,
    pretrained=True,
    momentum=0.9,
    weight_decay=0.001
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
