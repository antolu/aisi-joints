from self_supervised.model_params import BYOLParams

from self_supervised import LinearClassifierMethodParams

model_params = BYOLParams(
    dataset_name='aisi',
    encoder_arch='inception_resnet_v2',
    embedding_dim=1536,
    dim=2048,
    mlp_hidden_dim=1024,
    mlp_normalization='bn',
    lr=0.01,
    batch_size=64,
    lars_eta=0.01,
    lars_warmup_epochs=10,
    max_epochs=500,
    pretrained=False,
    optimizer_name='lars',
    momentum=0.9,
    use_momentum_schedule=True,
    weight_decay=0.005
)

classifier_params = LinearClassifierMethodParams(
    encoder_arch=model_params.encoder_arch,
    embedding_dim=model_params.embedding_dim,
    dataset_name=model_params.dataset_name,
    batch_size=64,
    lr=0.01,
    weight_decay=5.e-3,
    drop_last_batch=False,
    max_epochs=100,
    pretrained=False,
    class_weights={0: 0.86, 1: 1.2}
)
