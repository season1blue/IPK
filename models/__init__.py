from .tokenizer import Tokenizer

MODEL_NAMES = ['0_ours', '1_zipped', '2_full', '3_hierachical', '4_origin']

CLASS_MAPPER = {
    '0_ours': 'models.model_0_ours',
    '1_zipped': 'models.model_1_zipped',
    '2_full': 'models.model_2_full',
    '3_hierachical': 'models.model_3_hierachical',
    '4_origin': 'models.model_4_origin',
}
