from self_supervised.model_params import VICRegParams

from self_supervised import LinearClassifierMethodParams

model_params = VICRegParams(
    dataset_name="aisi",
    encoder_arch='inception_resnet_v2',
    shuffle_batch_norm=True,
    gather_keys_for_queue=True,
    # transform_apply_blur=False,
    mlp_hidden_dim=1024,
    dim=512,
    embedding_dim=1536,
    batch_size=64,
    lr=0.01,
    final_lr_schedule_value=0,
    weight_decay=0.01,
    momentum=0.9,
    lars_warmup_epochs=10,
    lars_eta=0.02,
    max_epochs=500,
    pretrained=False
)

classifier_params = LinearClassifierMethodParams(
    encoder_arch=model_params.encoder_arch,
    embedding_dim=model_params.embedding_dim,
    dataset_name=model_params.dataset_name,
    batch_size=64,
    lr=0.01,
    weight_decay=1.e-2,
    drop_last_batch=False,
    max_epochs=100,
    pretrained=False,
    class_weights={0: 0.86, 1: 1.2}
)
