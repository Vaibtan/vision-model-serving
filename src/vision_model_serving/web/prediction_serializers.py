"""Named, bounded serializers for the public prediction response contract."""

from rest_framework import serializers


class RequestEchoSerializer(serializers.Serializer):
    detector_score_threshold = serializers.FloatField(allow_null=True)


class InputSummarySerializer(serializers.Serializer):
    source_sha256 = serializers.CharField()
    rows = serializers.IntegerField()
    columns = serializers.IntegerField()
    frames = serializers.IntegerField()
    modality = serializers.CharField(allow_null=True)
    photometric_interpretation = serializers.CharField()
    presentation_lut_shape = serializers.CharField()
    transfer_syntax_uid = serializers.CharField()
    transfer_syntax_name = serializers.CharField()
    compressed = serializers.BooleanField()
    sop_class_uid = serializers.CharField(allow_null=True)
    bits_allocated = serializers.IntegerField(allow_null=True)
    bits_stored = serializers.IntegerField(allow_null=True)
    pixel_representation = serializers.IntegerField(allow_null=True)
    modality_transform_applied = serializers.BooleanField()
    voi_transform_applied = serializers.BooleanField()
    voi_index = serializers.IntegerField()


class GeometrySummarySerializer(serializers.Serializer):
    original_width = serializers.IntegerField()
    original_height = serializers.IntegerField()
    crop_box = serializers.ListField(child=serializers.IntegerField())
    canonical_width = serializers.IntegerField()
    canonical_height = serializers.IntegerField()
    scale_x = serializers.FloatField()
    scale_y = serializers.FloatField()


class DetectionSerializer(serializers.Serializer):
    rank = serializers.IntegerField()
    query_index = serializers.IntegerField()
    class_index = serializers.IntegerField()
    raw_logit = serializers.FloatField()
    score = serializers.FloatField()
    normalized_cxcywh = serializers.ListField(child=serializers.FloatField())
    normalized_xyxy = serializers.ListField(child=serializers.FloatField())
    canonical_xyxy = serializers.ListField(child=serializers.FloatField())
    original_xyxy = serializers.ListField(child=serializers.FloatField())
    padded = serializers.BooleanField()
    duplicate_of_rank = serializers.IntegerField(allow_null=True)


class DetectorInputSerializer(serializers.Serializer):
    source_size = serializers.ListField(child=serializers.IntegerField())
    resized_size = serializers.ListField(child=serializers.IntegerField())
    resize_short_edge = serializers.IntegerField()
    resize_max_edge = serializers.IntegerField()
    normalization_mean = serializers.ListField(child=serializers.FloatField())
    normalization_std = serializers.ListField(child=serializers.FloatField())


class DetectorPredictionSerializer(serializers.Serializer):
    input = DetectorInputSerializer()
    prediction_sha256 = serializers.CharField()
    top_candidates = DetectionSerializer(many=True)
    post_nms = DetectionSerializer(many=True)
    classifier_rois = DetectionSerializer(many=True)


class ClassifierInputSerializer(serializers.Serializer):
    label_information_used = serializers.BooleanField()
    token_count = serializers.IntegerField()
    clinical_text_truncated = serializers.BooleanField()
    crop_tensor_sha256 = serializers.CharField()
    input_ids_sha256 = serializers.CharField()
    attention_mask_sha256 = serializers.CharField()


class AttentionInspectionSerializer(serializers.Serializer):
    kind = serializers.CharField()
    roi_weights = serializers.ListField(child=serializers.FloatField())


class ClassificationPredictionSerializer(serializers.Serializer):
    input = ClassifierInputSerializer()
    class_indices = serializers.ListField(child=serializers.IntegerField())
    logits = serializers.ListField(child=serializers.FloatField())
    probabilities = serializers.ListField(child=serializers.FloatField())
    predicted_class_index = serializers.IntegerField()
    attention = AttentionInspectionSerializer()
    prediction_sha256 = serializers.CharField()


class ArtifactProvenanceSerializer(serializers.Serializer):
    id = serializers.CharField()
    sha256 = serializers.CharField()
    repository_revision = serializers.CharField()


class TokenizerProvenanceSerializer(serializers.Serializer):
    id = serializers.CharField()
    revision = serializers.CharField()
    file_sha256 = serializers.ListField(child=serializers.ListField(child=serializers.CharField()))


class PredictionProvenanceSerializer(serializers.Serializer):
    detector = ArtifactProvenanceSerializer()
    classifier = ArtifactProvenanceSerializer(allow_null=True)
    tokenizer = TokenizerProvenanceSerializer(allow_null=True)
    precision = serializers.CharField(allow_null=True)
    offline_assets_only = serializers.BooleanField(allow_null=True)
    strict_checkpoint_load = serializers.BooleanField(allow_null=True)


class MemorySummarySerializer(serializers.Serializer):
    allocated_bytes = serializers.IntegerField()
    reserved_bytes = serializers.IntegerField()
    peak_allocated_bytes = serializers.IntegerField()
    peak_reserved_bytes = serializers.IntegerField()


class RuntimeExecutionSerializer(serializers.Serializer):
    reused = serializers.BooleanField()
    load_ms = serializers.FloatField()
    inference_ms = serializers.FloatField()
    switch_ms = serializers.FloatField()


class DetectorAdapterTimingsSerializer(serializers.Serializer):
    load_ms = serializers.FloatField()
    preprocess_ms = serializers.FloatField()
    inference_ms = serializers.FloatField()
    postprocess_ms = serializers.FloatField()


class ClassifierAdapterTimingsSerializer(serializers.Serializer):
    load_ms = serializers.FloatField()
    crop_preprocess_ms = serializers.FloatField()
    tokenization_ms = serializers.FloatField()
    inference_ms = serializers.FloatField()
    result_ms = serializers.FloatField()


class DetectorStageTimingsSerializer(serializers.Serializer):
    runtime = RuntimeExecutionSerializer()
    adapter = DetectorAdapterTimingsSerializer()
    memory = MemorySummarySerializer()


class ClassifierStageTimingsSerializer(serializers.Serializer):
    runtime = RuntimeExecutionSerializer()
    adapter = ClassifierAdapterTimingsSerializer()
    memory = MemorySummarySerializer()


class PredictionTimingsSerializer(serializers.Serializer):
    decode_ms = serializers.FloatField()
    detector = DetectorStageTimingsSerializer()
    classifier = ClassifierStageTimingsSerializer(allow_null=True)
    pipeline_ms = serializers.FloatField()


class PredictionWarningSerializer(serializers.Serializer):
    stage = serializers.CharField()
    code = serializers.CharField()
    detail = serializers.CharField()


class PredictionResultSerializer(serializers.Serializer):
    mode = serializers.ChoiceField(choices=("detection", "full"))
    input = InputSummarySerializer()
    geometry = GeometrySummarySerializer()
    detector = DetectorPredictionSerializer()
    classification = ClassificationPredictionSerializer(allow_null=True)
    provenance = PredictionProvenanceSerializer()
    timings = PredictionTimingsSerializer()
    warnings = PredictionWarningSerializer(many=True)
    disclaimer = serializers.CharField()


class PredictionHandleResponseSerializer(serializers.Serializer):
    prediction_id = serializers.CharField()
    state = serializers.CharField()
    submitted_at = serializers.DateTimeField()
    expires_at = serializers.DateTimeField()
    status_url = serializers.CharField()
    result_url = serializers.CharField()
    idempotent_replay = serializers.BooleanField()
    request = RequestEchoSerializer()


class PredictionResultResponseSerializer(serializers.Serializer):
    prediction_id = serializers.CharField()
    result = PredictionResultSerializer()
    request = RequestEchoSerializer()


class PredictionSyncResponseSerializer(PredictionResultResponseSerializer):
    pass
