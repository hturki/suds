from enum import Enum


class SUDSFieldHeadNames(Enum):
    FEATURES = 'features'
    SHADOWS = 'shadows'
    BACKWARD_FLOW = 'backward_flow'
    FORWARD_FLOW = 'forward_flow'
    DYNAMIC_WEIGHT = 'dynamic_weight'

    BACKWARD_RGB = 'backward_rgb'
    BACKWARD_DENSITY = 'backward_density'
    BACKWARD_FEATURES = 'backward_features'
    BACKWARD_FLOW_CYCLE_DIFF = 'backward_flow_cycle_diff'
    BACKWARD_DYNAMIC_WEIGHT = 'backward_dynamic_weight'

    FORWARD_RGB = 'forward_rgb'
    FORWARD_DENSITY = 'forward_density'
    FORWARD_FEATURES = 'forward_features'
    FORWARD_FLOW_CYCLE_DIFF = 'forward_flow_cycle_diff'
    FORWARD_DYNAMIC_WEIGHT = 'forward_dynamic_weight'

    FLOW_SLOW = 'flow_slow'
    FLOW_SMOOTH_TEMPORAL = 'flow_smooth_temporal'

    NO_SHADOW_RGB = 'no_shadow_rgb'
