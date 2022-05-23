from self_supervised import LinearClassifierMethodParams
from self_supervised.model_params import VICRegParams


model_params = VICRegParams(
    dataset_name="aisi",
    encoder_arch='resnet101',
    shuffle_batch_norm=True,
    gather_keys_for_queue=True,
    # transform_apply_blur=False,
    mlp_hidden_dim=2048,
    dim=2048,
    embedding_dim=2048,
    batch_size=32,
    lr=0.03,
    final_lr_schedule_value=0,
    weight_decay=1e-4,
    lars_warmup_epochs=10,
    lars_eta=0.02
)

classifier_params = LinearClassifierMethodParams(
    encoder_arch=model_params.encoder_arch,
    embedding_dim=model_params.embedding_dim,
    dataset_name=model_params.dataset_name,
    batch_size=32,
    lr=10,
    weight_decay=1e-4,
    drop_last_batch=False
)
