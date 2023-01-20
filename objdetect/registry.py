from detectron2.utils.registry import Registry

PROPOSAL_REGISTRY = Registry("PROPOSAL_REGISTRY")
PROPOSAL_REGISTRY.__doc__ = """
Registry for proposals generation used in prox models e.g. uniform, perturbation of ground truth

The registered object must be a callable that accepts one argument:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`bool`, which represents whether the input is a validation proposal.

Registered object must return instance of :class:`nn.Module`.
"""

ENCODER_REGISTRY = Registry("ENCODER_REGISTRY")
ENCODER_REGISTRY.__doc__ = """
Registry for encoders used in prox models.

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.

Registered object must return instance of :class:`nn.Module`.
"""


NETWORK_REGISTRY = Registry("NETWORK_REGISTRY")
NETWORK_REGISTRY.__doc__ = """
Registry for networks used in prox models after the encoder, which operates on the output of e.g. resnet

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.

Registered object must return instance of :class:`nn.Module`.
"""


LOSS_REGISTRY = Registry("LOSS_REGISTRY")
LOSS_REGISTRY.__doc__ = """
Registry for losses used in prox models e.g. Hungarian, Projection

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.

Registered object must return instance of :class:`nn.Module`.
"""
