from self_supervised import LinearClassifierMethodParams
from self_supervised.model_params import VICRegParams, ModelParams, BYOLParams

model_params = VICRegParams(
    dataset_name="aisi",
    encoder_arch='inception_resnet_v2',
    shuffle_batch_norm=True,
    gather_keys_for_queue=True,
    # transform_apply_blur=False,
    mlp_hidden_dim=2048,
    dim=2048,
    embedding_dim=1536,
    batch_size=32,
    lr=0.01,
    final_lr_schedule_value=0,
    weight_decay=1e-4,
    lars_warmup_epochs=10,
    lars_eta=0.02,
    max_epochs=200,
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
